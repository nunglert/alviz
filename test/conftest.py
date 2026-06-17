import matplotlib
import pytest

# Headless backend so the suite renders without a display.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test opened (each Dashboard creates one).

    Dashboards hold their figure open until closed; without this the suite
    leaks figures across tests and matplotlib warns past 20 open.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
