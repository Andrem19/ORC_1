"""Plan-mode orchestrator service — composition via mixins."""

from app.services.plan_orchestrator._core import PlanOrchestratorCore
from app.services.plan_orchestrator._planner_monitor import PlannerMonitorMixin
from app.services.plan_orchestrator._plan_lifecycle import PlanLifecycleMixin
from app.services.plan_orchestrator._task_health import TaskHealthMixin
from app.services.plan_orchestrator._task_dispatch import TaskDispatchMixin
from app.services.plan_orchestrator._result_processing import ResultProcessingMixin


class PlanOrchestratorService(
    PlannerMonitorMixin,
    PlanLifecycleMixin,
    TaskHealthMixin,
    TaskDispatchMixin,
    ResultProcessingMixin,
    PlanOrchestratorCore,
):
    """Runs the plan-driven orchestrator loop.

    Composed from domain-specific mixins.  See individual modules
    for implementation details.
    """
