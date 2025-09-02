import pytest
from odecast import t, var, Eq, solve, BC


@pytest.mark.xfail(reason="BVP arrives in Milestone 5")
def test_bvp_basic():
    y = var("y")
    eq = Eq(y.d(2) + y, 0)
    sol = solve(
        eq,
        bvp=[BC(y, t=0, value=0), BC(y, t=1, value=1)],
        tspan=(0, 1),
        backend="scipy_bvp",
    )
    assert sol[y].shape[0] > 2
