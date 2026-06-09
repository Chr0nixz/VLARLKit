from vlarlkit.rollouts.rollout import Rollout

__all__ = ["Rollout", "BranchRollout"]


def __getattr__(name: str):
    if name == "BranchRollout":
        from vlarlkit.rollouts.branch_rollout import BranchRollout

        return BranchRollout
    raise AttributeError(name)
