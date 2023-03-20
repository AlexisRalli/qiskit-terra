# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Multiple-Controlled U3 gate. Not using ancillary qubits.
"""

from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
from qiskit.circuit._utils import _ctrl_state_to_int, _compute_control_matrix
from qiskit.circuit.controlledgate import ControlledGate


def _apply_cu(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    if use_basis_gates:
        # pylint: disable=cyclic-import
        #          ┌──────────────┐
        # control: ┤ P(λ/2 + φ/2) ├──■──────────────────────────────────■────────────────
        #          ├──────────────┤┌─┴─┐┌────────────────────────────┐┌─┴─┐┌────────────┐
        #  target: ┤ P(λ/2 - φ/2) ├┤ X ├┤ U(-0.5*0,0,-0.5*λ - 0.5*φ) ├┤ X ├┤ U(0/2,φ,0) ├
        #          └──────────────┘└───┘└────────────────────────────┘└───┘└────────────┘
        circuit.p((lam + phi) / 2, [control])
        circuit.p((lam - phi) / 2, [target])
        circuit.cx(control, target)
        circuit.u(-theta / 2, 0, -(phi + lam) / 2, [target])
        circuit.cx(control, target)
        circuit.u(theta / 2, phi, 0, [target])
    else:
        circuit.cu(theta, phi, lam, 0, control, target)


def _apply_mcu_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    """Apply multi-controlled u gate from ctls to tgt using graycode
    pattern with single-step angles theta, phi, lam."""

    n = len(ctls)

    gray_code = _generate_gray_code(n)
    last_pattern = None

    for pattern in gray_code:
        if "1" not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index("1")

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == "1"]
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        # check parity and undo rotation
        if pattern.count("1") % 2 == 0:
            # inverse CU: u(theta, phi, lamb)^dagger = u(-theta, -lam, -phi)
            _apply_cu(
                circuit, -theta, -lam, -phi, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates
            )
        else:
            _apply_cu(circuit, theta, phi, lam, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        last_pattern = pattern


def mcsu2_real_diagonal(
    circuit,
    unitary: np.ndarray,
    controls: Union[QuantumRegister, List[Qubit]],
    target: Union[Qubit, int],
    ctrl_state: str = None,
):
    """
    Apply multi-controlled SU(2) gate with one real diagonal.
    https://arxiv.org/abs/2302.06377

    Args:
        circuit (QuantumCircuit): The QuantumCircuit object to apply the diagonal operator on.
        unitary (ndarray): SU(2) unitary matrix with one real diagonal
        controls (QuantumRegister or list(Qubit)): The list of control qubits
        target (Qubit or int): The target qubit
        ctrl_state (str): control state of the operator SU(2) operator

    Raises:
        QiskitError: parameter errors
    """
    # pylint: disable=cyclic-import
    from qiskit.circuit.library import MCXVChain
    from qiskit.extensions import UnitaryGate
    from qiskit.quantum_info.operators.predicates import is_unitary_matrix

    if not is_unitary_matrix(unitary):
        raise QiskitError("parameter unitary in linear_depth_mcv must be an unitary matrix")

    if unitary.shape != (2, 2):
        raise QiskitError("parameter unitary in linear_depth_mcv must be a 2x2 matrix")

    is_main_diag_real = np.isclose(unitary[0, 0].imag, 0.0) and np.isclose(unitary[1, 1].imag, 0.0)
    is_secondary_diag_real = np.isclose(unitary[0, 1].imag, 0.0) and np.isclose(
        unitary[1, 0].imag, 0.0
    )

    if not is_main_diag_real and not is_secondary_diag_real:
        raise QiskitError("parameter unitary in linear_depth_mcv must have one real diagonal")

    if is_secondary_diag_real:
        x = unitary[0, 1]
        z = unitary[1, 1]
    else:
        x = -unitary[0, 1].real
        z = unitary[1, 1] - unitary[0, 1].imag * 1.0j

    alpha_r = np.sqrt((np.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
    alpha_i = z.imag / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))
    alpha = alpha_r + 1.0j * alpha_i
    beta = x / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))

    # S gate definition
    s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])
    s_gate = UnitaryGate(s_op)

    num_ctrl = len(controls)
    k_1 = int(np.ceil(num_ctrl / 2.0))
    k_2 = int(np.floor(num_ctrl / 2.0))

    ctrl_state_k_1 = None
    ctrl_state_k_2 = None

    if ctrl_state is not None:
        str_ctrl_state = f"{ctrl_state:0{num_ctrl}b}"
        ctrl_state_k_1 = str_ctrl_state[::-1][:k_1][::-1]
        ctrl_state_k_2 = str_ctrl_state[::-1][k_1:][::-1]

    if not is_secondary_diag_real:
        circuit.h(target)

    mcx_1 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_1, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.append(s_gate, [target])

    mcx_2 = MCXVChain(
        num_ctrl_qubits=k_2,
        dirty_ancillas=True,
        ctrl_state=ctrl_state_k_2,
        # action_only=general_su2_optimization # Requires PR #9687
    )
    circuit.append(mcx_2.inverse(), controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.append(s_gate.inverse(), [target])

    mcx_3 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_3, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.append(s_gate, [target])

    mcx_4 = MCXVChain(num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2)
    circuit.append(mcx_4, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.append(s_gate.inverse(), [target])

    if not is_secondary_diag_real:
        circuit.h(target)


def mcrx(
    self,
    theta: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    q_ancillae: Optional[Union[QuantumRegister, Tuple[QuantumRegister, int]]] = None,
    mode: str = None,
    use_basis_gates: bool = False,
):
    """
    Apply Multiple-Controlled X rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrx gate on.
        theta (float): angle theta
        q_controls (QuantumRegister or list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancillae (QuantumRegister or tuple(QuantumRegister, int)): The list of ancillary qubits.
        mode (string): The implementation mode to use
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    ancillary_qubits = [] if q_ancillae is None else self.qbit_argument_conversion(q_ancillae)
    all_qubits = control_qubits + target_qubit + ancillary_qubits
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)

    # auto-select the best mode
    if mode is None:
        # if enough ancillary qubits are provided, use the 'v-chain' method
        additional_vchain = MCXGate.get_num_ancilla_qubits(len(control_qubits), "v-chain")
        if len(ancillary_qubits) >= additional_vchain:
            mode = "basic"
        else:
            mode = "noancilla"

    if mode == "basic":
        self.rx(theta / 2, q_target)

        self.h(q_target)
        self.mcx(q_controls, q_target, ancillary_qubits, mode="v-chain")
        self.h(q_target)

        self.rx(-theta / 2, q_target)

        self.h(q_target)
        self.mcx(q_controls, q_target, ancillary_qubits, mode="v-chain")
        self.h(q_target)
    elif mode == "noancilla":
        n_c = len(control_qubits)
        rot_gate = ControlRotationGate(
            theta, n_c, "x", ctrl_state=None, use_basis_gates=use_basis_gates
        )
        self.append(rot_gate, [*q_controls, target_qubit])
    else:
        raise QiskitError(f"Unrecognized mode for building MCRY circuit: {mode}.")


def mcry(
    self,
    theta: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    q_ancillae: Optional[Union[QuantumRegister, Tuple[QuantumRegister, int]]] = None,
    mode: str = None,
    use_basis_gates=False,
):
    """
    Apply Multiple-Controlled Y rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcry gate on.
        theta (float): angle theta
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancillae (QuantumRegister or tuple(QuantumRegister, int)): The list of ancillary qubits.
        mode (string): The implementation mode to use
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    ancillary_qubits = [] if q_ancillae is None else self.qbit_argument_conversion(q_ancillae)
    all_qubits = control_qubits + target_qubit + ancillary_qubits
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)

    # auto-select the best mode
    if mode is None:
        # if enough ancillary qubits are provided, use the 'v-chain' method
        additional_vchain = MCXGate.get_num_ancilla_qubits(len(control_qubits), "v-chain")
        if len(ancillary_qubits) >= additional_vchain:
            mode = "basic"
        else:
            mode = "noancilla"

    if mode == "basic":
        self.ry(theta / 2, q_target)
        self.mcx(q_controls, q_target, ancillary_qubits, mode="v-chain")
        self.ry(-theta / 2, q_target)
        self.mcx(q_controls, q_target, ancillary_qubits, mode="v-chain")
    elif mode == "noancilla":
        n_c = len(control_qubits)
        rot_gate = ControlRotationGate(
            theta, n_c, "y", ctrl_state=None, use_basis_gates=use_basis_gates
        )
        self.append(rot_gate, [*q_controls, target_qubit])
    else:
        raise QiskitError(f"Unrecognized mode for building MCRY circuit: {mode}.")


def mcrz(
    self,
    lam: ParameterValueType,
    q_controls: Union[QuantumRegister, List[Qubit]],
    q_target: Qubit,
    use_basis_gates: bool = False,
):
    """
    Apply Multiple-Controlled Z rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrz gate on.
        lam (float): angle lambda
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use p, u, cx

    Raises:
        QiskitError: parameter errors
    """
    control_qubits = self.qbit_argument_conversion(q_controls)
    target_qubit = self.qbit_argument_conversion(q_target)
    if len(target_qubit) != 1:
        raise QiskitError("The mcrz gate needs a single qubit as target.")
    all_qubits = control_qubits + target_qubit
    target_qubit = target_qubit[0]
    self._check_dups(all_qubits)

    n_c = len(control_qubits)
    if n_c == 1:  # cu
        _apply_cu(self, 0, 0, lam, control_qubits[0], target_qubit, use_basis_gates=use_basis_gates)
    else:
        lam_step = lam * (1 / (2 ** (n_c - 1)))
        _apply_mcu_graycode(
            self, 0, 0, lam_step, control_qubits, target_qubit, use_basis_gates=use_basis_gates
        )


class ControlRotationGate(ControlledGate):
    """
    Multicontrol single qubit Pauli rotation gate.

    """

    def __init__(
        self,
        angle: ParameterValueType,
        num_ctrl_qubits: int,
        axis: str,
        ctrl_state: Optional[Union[str, int]] = None,
        use_basis_gates: bool = False,
    ):
        """
        Multicontrol single qubit Pauli rotation gate.

        6 controls and below uses quantum multiplexors: see Theorem 8 of https://arxiv.org/pdf/quant-ph/0406176.pdf
        above 6 qubits the decomposition uses multi-controlled SU(2) strategy given in https://arxiv.org/abs/2302.06377

        Args:
            angle (float): angle of rotation
            ctrl_list (list): list of control qubit indices
            target (int): index of target qubit
            axis (str): x,y,z axis
            use_basis_gates (bool): use p, u, cx

        """

        assert axis in ["x", "y", "z"], f"can only rotated around x,y,z axis, not {axis}"

        self.axis = axis
        self.angle = angle
        self.use_basis_gates = use_basis_gates

        if self.axis == "x":
            self.base_gate = RXGate(angle)
        elif self.axis == "y":
            self.base_gate = RYGate(angle)
        elif self.axis == "z":
            self.base_gate = RZGate(angle)

        self._num_qubits = num_ctrl_qubits + 1
        self.num_ctrl_qubits = num_ctrl_qubits

        cntrl_int = _ctrl_state_to_int(ctrl_state, self.num_ctrl_qubits)
        cntrl_str = np.binary_repr(cntrl_int, width=self.num_ctrl_qubits)
        self.ctrl_state = cntrl_str[::-1]

        self.label = self.base_gate.label
        super().__init__(
            name=self.label,
            num_qubits=self._num_qubits,
            params=[angle],
            num_ctrl_qubits=self.num_ctrl_qubits,
            ctrl_state=self.ctrl_state,
            base_gate=self.base_gate,
        )

    def _define(self):
        """
        define gate by a circuit decomposition
        """
        cntrl_int = _ctrl_state_to_int(self.ctrl_state, self.num_ctrl_qubits)
        self.definition = QuantumCircuit(self.num_ctrl_qubits + 1)

        if self.num_ctrl_qubits < 6:

            angle_list = [0] * 2 ** (self.num_ctrl_qubits)
            angle_list[cntrl_int] = self.angle

            if self.axis == "x":
                self.definition.ucrx(
                    angle_list, list(range(self.num_ctrl_qubits)), self.num_ctrl_qubits
                )
            elif self.axis == "y":
                self.definition.ucry(
                    angle_list, list(range(self.num_ctrl_qubits)), self.num_ctrl_qubits
                )
            elif self.axis == "z":
                self.definition.ucrz(
                    angle_list, list(range(self.num_ctrl_qubits)), self.num_ctrl_qubits
                )
        else:
            if self.axis == "x":
                mcsu2_real_diagonal(
                    self.definition,
                    RXGate(self.angle).to_matrix(),
                    list(range(self.num_ctrl_qubits)),
                    self.num_ctrl_qubits,
                    ctrl_state=self.ctrl_state,
                )
            elif self.axis == "y":
                mcsu2_real_diagonal(
                    self.definition,
                    RYGate(self.angle).to_matrix(),
                    list(range(self.num_ctrl_qubits)),
                    self.num_ctrl_qubits,
                    ctrl_state=self.ctrl_state,
                )
            elif self.axis == "z":
                mcsu2_real_diagonal(
                    self.definition,
                    RZGate(self.angle).to_matrix(),
                    list(range(self.num_ctrl_qubits)),
                    self.num_ctrl_qubits,
                    ctrl_state=self.ctrl_state,
                )

        if self.use_basis_gates:
            # unroll into basis gates
            from qiskit.transpiler.passes import Unroller
            from qiskit.transpiler import PassManager

            pass_ = Unroller(["p", "u", "cx"])
            pm = PassManager(pass_)
            self.definition = pm.run(self.definition)

    def inverse(self):
        """
        Returns inverse rotation gate
        """
        return ControlRotationGate(
            -1 * self.angle,
            self.num_ctrl_qubits,
            self.axis,
            ctrl_state=self.ctrl_state,
            use_basis_gates=self.use_basis_gates,
        )

    def __array__(self, dtype=None):
        """
        Return numpy array for gate
        """
        mat = _compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        if dtype:
            mat = np.asarray(mat, dtype=dtype)
        return mat


QuantumCircuit.mcrx = mcrx
QuantumCircuit.mcry = mcry
QuantumCircuit.mcrz = mcrz
