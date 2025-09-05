import verifiers as vf
import vf_exts as vfe

def load_environment(
    num_train_examples: int = -1, num_eval_examples: int = -1
) -> vfe.MultiTurnEnvGroup:
    envs = [
        vf.load_environment("vf-gsm8k", num_train_examples=5000, num_eval_examples=num_eval_examples),
        vf.load_environment("vf-ifeval", num_train_examples=5000, num_eval_examples=num_eval_examples),
        vf.load_environment("vf-basicqa", num_train_examples=5000, num_eval_examples=num_eval_examples),
        vf.load_environment("vf-grouped", num_train_examples=10000, num_eval_examples=num_eval_examples),
        vf.load_environment("vf-codestd", num_train_examples=10000, num_eval_examples=num_eval_examples),
    ]
    env_names = [
        "vf-gsm8k",
        "vf-ifeval",
        "vf-basicqa",
        "vf-grouped",
        "vf-codestd",
    ]
    return vfe.MultiTurnEnvGroup(envs, env_names=env_names)