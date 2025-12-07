"""Package info for vf_exts."""

MAJOR = 2
MINOR = 0
PATCH = 0
PRE_RELEASE = "dev"

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "vf_exts"
__contact_names__ = "Arcee AI"
__contact_emails__ = "aria@arcee.ai"
__homepage__ = "https://github.com/arcee-ai/RLKit"
__repository_url__ = "https://github.com/arcee-ai/RLKit"
__download_url__ = "https://github.com/arcee-ai/RLKit/releases"
__description__ = "RLKit vf_exts - verifiers environment stdlib for RLKit"
__license__ = "Apache2"
__keywords__ = "reinforcement learning, RLKit"
