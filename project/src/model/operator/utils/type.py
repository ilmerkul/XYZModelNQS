from ..operators import (
    create_polynomial_operator,
    create_shift_invert_operator,
    create_shifted_operator,
)
from ..interface import PreconditionerType

type2operator = {
            PreconditionerType.Exact: None,
            PreconditionerType.Polynominal: create_polynomial_operator,
            PreconditionerType.ShiftInvert: create_shift_invert_operator,
            PreconditionerType.Shift: create_shifted_operator,
        }