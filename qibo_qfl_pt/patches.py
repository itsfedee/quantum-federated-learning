"""Monkey-patches for qiboml and qibo."""

import numpy as np
import torch
import qibo.models.error_mitigation as _em
import qiboml.interfaces.pytorch as pt
from functools import reduce


# =====================================================================
# Patch 1: deterministic CDR
# sample_training_circuit_cdr calls backend.set_seed(None) on every
# iteration, which reseeds np.random at random and destroys
# reproducibility. The fix installs a temporary guard that ignores the
# np.random.seed(None) calls while the CDR is running.
# =====================================================================

# Permanent guard: stops backend.set_seed(None) from randomising
# np.random. It is needed because several qibo functions
# (sample_training_circuit_cdr, error_sensitive_circuit, and others) call
# backend.set_seed(None), which reseeds np.random at random.
_original_np_seed = np.random.seed


def _guarded_seed(seed):
    """Ignore np.random.seed(None), to avoid random resets of the RNG."""
    if seed is not None:
        _original_np_seed(seed)


np.random.seed = _guarded_seed

# Patch 1b: CDR isolated from the backend RNG
# CDR calls np.random.seed(40) but does not reset qibo's backend RNG,
# which depends on the external seed of the run. That introduces variance
# between runs with different seeds. The fix resets qibo's backends too,
# before the CDR.
_original_CDR = _em.CDR


def _isolated_CDR(*args, seed=None, **kwargs):
    if seed is not None:
        _original_np_seed(seed)
        _em.SIMULATION_BACKEND().set_seed(seed)
        _em.CLIFFORD_BACKEND().set_seed(seed)
    return _original_CDR(*args, seed=seed, **kwargs)


_em.CDR = _isolated_CDR


# =====================================================================
# Patch 2: QuantumModelAutoGrad fix
# A fix for the PSR, which does not handle the parameters correctly.
# =====================================================================

def _get_angles(circuit, include_not_trainable):
    return np.array(
        [float(par)
         for params in circuit.get_parameters(include_not_trainable=include_not_trainable)
         for par in params],
        dtype=np.float32,
    )


class _FixedQuantumModelAutoGrad(torch.autograd.Function):

    # in the forward pass we split the trainable parameters from the total
    # ones (trainable + encoding)
    @staticmethod
    def forward(ctx, x, decoding, differentiation, circuit_tracer, *parameters):
        parameters = torch.stack(parameters)
        circuit, jacobian_wrt_inputs, jacobian, input_to_gate_map = circuit_tracer(
            parameters, x=x
        )
        dtype = getattr(decoding.backend.np, str(parameters.dtype).split(".")[-1])

        all_angles = decoding.backend.cast(_get_angles(circuit, True), dtype=dtype) # tutti
        for g, p in zip(differentiation.circuit.parametrized_gates, all_angles):
            g.parameters = p

        trainable_angles = decoding.backend.cast(_get_angles(circuit, False), dtype=dtype) # trainable

        ctx.save_for_backward(jacobian_wrt_inputs, jacobian)
        ctx.angles = trainable_angles # trainable
        ctx.all_angles = all_angles # tutti
        ctx.differentiation = differentiation
        ctx.input_to_gate_map = input_to_gate_map
        ctx.dtype = dtype
        ctx.wrt_inputs = jacobian_wrt_inputs is not None

        x_out = decoding(differentiation.circuit)
        del circuit
        x_out = torch.as_tensor(
            decoding.backend.to_numpy(x_out).tolist(),
            dtype=parameters.dtype,
            device=parameters.device,
        )
        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        jacobian_wrt_inputs, jacobian = ctx.saved_tensors
        backend = ctx.differentiation.decoding.backend

        # fix: restore the circuit state for this sample.
        # differentiation.circuit is shared: other forward passes may have
        # overwritten its angles. PSR uses the current state as the unshifted
        # baseline, so without restoring it we would measure from the wrong point.
   
        for g, p in zip(ctx.differentiation.circuit.parametrized_gates, ctx.all_angles):
            g.parameters = p

        angles_to_pass = ctx.all_angles if ctx.wrt_inputs else ctx.angles
        psr_result = ctx.differentiation.evaluate(angles_to_pass, wrt_inputs=ctx.wrt_inputs)
        jacobian_wrt_angles = torch.as_tensor(
            backend.to_numpy(psr_result),
            dtype=jacobian.dtype,
            device=jacobian.device,
        )
        del psr_result

        out_shape = ctx.differentiation.decoding.output_shape
        contraction = ((0, 1), (0,) + tuple(range(2, len(out_shape) + 2)))
        right_indices = tuple(range(1, len(grad_output.shape) + 1))
        left_indices = (0,) + right_indices

        if jacobian_wrt_inputs is not None:
            jacobian_wrt_encoding_angles = torch.vstack(
                [jacobian_wrt_angles[list(indices)]
                 for indices in zip(*ctx.input_to_gate_map.values())]
            )
            indices_to_discard = reduce(tuple.__add__, ctx.input_to_gate_map.values())
            jacobian_wrt_angles = torch.vstack(
                [row for i, row in enumerate(jacobian_wrt_angles)
                 if i not in indices_to_discard]
            ).reshape(-1, *out_shape)
            grad_input = torch.einsum(
                jacobian_wrt_inputs, contraction[0],
                jacobian_wrt_encoding_angles, contraction[1],
            )
            grad_input = torch.einsum(grad_input, left_indices, grad_output, right_indices)
        else:
            grad_input = None

        gradient = torch.einsum(
            jacobian, contraction[0], jacobian_wrt_angles, contraction[1]
        )
        gradient = torch.einsum(gradient, left_indices, grad_output, right_indices)
        return (grad_input, None, None, None, *gradient)


pt.QuantumModelAutoGrad = _FixedQuantumModelAutoGrad


# =====================================================================
# Patch 3: injecting the CDR parameters (warm start)
# qiboml exposes no public method to inject (a, b) from outside. This
# function sets the internal fields of the mitigator so that the first
# forward pass does NOT trigger the automatic calibration.
# =====================================================================

def inject_cdr_map(model, a, b):
    """Inject the CDR parameters (a, b) into the mitigator of a QMLModel.

    Dopo l'iniezione il mitigator si comporta come se avesse già calibrato:
    usa la mappa iniettata e ricalibra solo quando la soglia RTQEM lo richiede.

    Il reference_value viene calcolato al primo forward (costa poco, è noiseless)
    tramite il flusso standard di qiboml — noi settiamo solo (a, b).
    """
    mitigator = getattr(
        getattr(getattr(model, "q_model", None), "decoding", None),
        "mitigator", None,
    )
    if mitigator is None:
        return False

    backend = mitigator.backend
    popt = (float(a), float(b))
    mitigator._mitigation_map_popt = backend.cast(popt, dtype="double")
    mitigator._mitigation_map.__defaults__ = popt
    return True


# =====================================================================
# Patch 4: RTQEM diagnostic logging
# Records the deviation D at every threshold check, without changing the
# decision. It tells whether the recalibrations come from shot noise or
# from a systematic bias.
# =====================================================================

from qiboml.models.utils import Mitigator as _Mitigator

_original_map_is_reliable = _Mitigator.map_is_reliable


def _logged_map_is_reliable(self, noisy_reference_value):
    mitigated_ref = self._mitigation_map(noisy_reference_value, *self._mitigation_map_popt)
    D = float(abs(mitigated_ref - self._reference_value))
    if not hasattr(self, "_check_log"):
        self._check_log = []
    self._check_log.append({
        "step": getattr(self, "_n_checks", 0),
        "D": D,
        "noisy_ref": float(noisy_reference_value),
        "exact_ref": float(self._reference_value),
        "a": float(self._mitigation_map_popt[0]),
        "b": float(self._mitigation_map_popt[1]),
        "failed": D > self._threshold,
    })
    return _original_map_is_reliable(self, noisy_reference_value)


_Mitigator.map_is_reliable = _logged_map_is_reliable


# =====================================================================
# Patch 5: intra-round EMA on the data_regression
# When the mitigator recalibrates, instead of overwriting (a, b) with the
# fresh fit it blends them with an EMA: a <- beta*a_old + (1-beta)*a_new.
# Turned on through mitigation_config: "with_memory": True,
# "memory_beta": 0.3, or by setting mitigator._intra_ema_beta directly.
# =====================================================================

_original_post_init = _Mitigator.__post_init__


def _patched_post_init(self):
    _original_post_init(self)
    cfg = self.mitigation_config or {}
    if cfg.get("with_memory", False):
        self._intra_ema_beta = cfg.get("memory_beta", 0.5)
    # Can be overridden from outside (client_app, for instance)


_Mitigator.__post_init__ = _patched_post_init

_original_data_regression = _Mitigator.data_regression


def _ema_data_regression(self, *args, **kwargs):
    beta = getattr(self, "_intra_ema_beta", None)

    # Save the old popt before the refit
    old_popt = None
    if beta is not None and beta > 0:
        prev = getattr(self, "_mitigation_map_popt", None)
        if prev is not None:
            old_popt = tuple(float(p) for p in prev)

    # Call the original refit (it overwrites _mitigation_map_popt)
    _original_data_regression(self, *args, **kwargs)

    # Apply the EMA when enabled and when a previous non-identity map existed
    if old_popt is not None and not (old_popt[0] == 1.0 and old_popt[1] == 0.0):
        new_a = float(self._mitigation_map_popt[0])
        new_b = float(self._mitigation_map_popt[1])
        ema_a = beta * old_popt[0] + (1 - beta) * new_a
        ema_b = beta * old_popt[1] + (1 - beta) * new_b
        popt = (ema_a, ema_b)
        self._mitigation_map.__defaults__ = popt
        self._mitigation_map_popt = self.backend.cast(popt, dtype="double")


_Mitigator.data_regression = _ema_data_regression
