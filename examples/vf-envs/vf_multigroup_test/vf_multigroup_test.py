import verifiers as vf
import vf_exts as vfe

def load_environment(
    num_train_examples: int = 1000, num_eval_examples: int = 100
) -> vfe.MultiTurnEnvGroup:
    envs = [
        vf.load_environment("vf-reverse-text"),
        vf.load_environment("vf-tool-test"),
        vf.load_environment("vf-alphabet-sort")
    ]
    return vfe.MultiTurnEnvGroup(envs)