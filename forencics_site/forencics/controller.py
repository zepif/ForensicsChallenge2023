from dataclasses import dataclass
from threading import RLock

@dataclass
class State:
    placeholder: int


class Controller:
    """Main class for controlling the flow of the program"""

    def __init__(self):
        self.lock = RLock()

    def load(self):
        """Loads the config"""

    def user_uploaded_file(self, filename: str):
        """User uploads file"""
        # TODO
    
    def user_file_flash(self):
        """User wants to remove last file"""
        print("pepka")
        # TODO

    
    
    # def team_enter_code(self, team_id: str, code: str):
    #     """
    #     Team enters code

    #     Return:
    #         True if moved to next stage, else False
    #     """

    #     with self.lock:
    #         timestamp = time.time()

    #         self._log_event(
    #             "team_enter_code",
    #             team_id=team_id,
    #             code=code,
    #             timestamp=timestamp,
    #         )

    #         return self._team_enter_code(
    #             team_id=team_id,
    #             code=code,
    #             timestamp=timestamp,
    #         )

    # def _team_enter_code(self, team_id: str, code: str, timestamp: float):
    #     """Enter code logic"""

    #     team_info = self.team_info[team_id]

    #     if team_info.finished:
    #         return False

    #     stage = self.config.stages[team_info.stage]
    #     time_since_stage_entered = timestamp - team_info.stage_entered_timestamps[-1]

    #     code_accepted = False

    #     for task in stage.tasks:
    #         if task.code == code and task.id not in team_info.completed_tasks:
    #             points = self._get_points_for_completed_task(
    #                 task_config=task,
    #                 time_since_stage_entered=time_since_stage_entered,
    #             )

    #             team_info.total_points += points
    #             if task.counts_towards_total_time:
    #                 team_info.total_time += time_since_stage_entered

    #             team_info.completed_tasks[task.id] = TaskCompletionInfo(
    #                 timestamp=timestamp,
    #                 time_since_stage_entered=time_since_stage_entered,
    #                 points=points,
    #             )

    #             code_accepted = True

    #     message_text = f"Введено код: {code}. "
    #     if code_accepted:
    #         message_text += "Код зараховано!"
    #     else:
    #         message_text += "Неправильний код!"

    #     team_info.messages.append(
    #         Message(
    #             timestamp=timestamp,
    #             text=message_text,
    #         )
    #     )

    #     stage_upped = self.up_stage_if_needed(team_id=team_id, timestamp=timestamp)

    #     return stage_upped
