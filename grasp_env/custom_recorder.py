# simple_image_recorder.py

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import ActionStateRecorderManagerCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from collections.abc import Sequence
class ImageRecorder(RecorderTerm):
    """Simple recorder for image observations with frequency control."""
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.step_counter = 0
        self.record_frequency = 10  # Record every 50 steps
    
    def record_pre_step(self):
        
        
        # Only record every 50 steps
        if self.step_counter % self.record_frequency == 0 or self.step_counter == 3:
            # print(self.step_counter)
            self.step_counter += 1
            return "image_obs", self._env.obs_buf["Image_info"]
        self.step_counter += 1
        
        # Return None to skip recording this step
        return None, None
    def reset(self, env_ids):
        """Reset the step counter."""
        self.step_counter = 0
      
class TaskInfoRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)

    def record_post_reset(self, env_ids: Sequence[int] | None):

        return "task_info", self._env.cfg.task_info[env_ids]

@configclass
class CustomImageRecorderCfg(RecorderTermCfg):
    """Configuration for image recorder."""
    
    class_type: type[RecorderTerm] = ImageRecorder


@configclass
class TaskInfoRecorderCfg(RecorderTermCfg):
    """Configuration for task_info recorder."""
    class_type: type[RecorderTerm] = TaskInfoRecorder

@configclass
class MultiModalRecorderCfg(ActionStateRecorderManagerCfg):
    """Add image recording to existing recorder."""
    record_images = CustomImageRecorderCfg()
    # record_task_info = TaskInfoRecorderCfg()



    
