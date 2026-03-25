from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


class Tiltrotor6dLabEnv(ManagerBasedRLEnv):
    """Manager-based RL env with explicit termination-reason logging."""

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments and expose exact done causes."""
        term_counts: dict[str, float] = {}
        summary_items: list[str] = []

        if len(env_ids) > 0:
            num_reset_envs = int(len(env_ids))
            term_counts["Episode_Termination_Count/reset_envs"] = float(
                num_reset_envs
            )
            for term_name in self.termination_manager.active_terms:
                term_value = self.termination_manager.get_term(term_name)
                count = int(torch.count_nonzero(term_value[env_ids]).item())
                term_counts[
                    f"Episode_Termination_Count/{term_name}"
                ] = float(count)
                if count > 0:
                    summary_items.append(f"{term_name}={count}")

        super()._reset_idx(env_ids)

        if term_counts:
            self.extras.setdefault("log", {})
            self.extras["log"].update(term_counts)

        if summary_items:
            summary = ", ".join(summary_items)
            print(f"[Termination] reset {int(len(env_ids))} env(s): {summary}")
