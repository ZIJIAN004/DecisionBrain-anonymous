"""DecisionBrain project exceptions independent of web frameworks."""


class ORAgentError(Exception):
    """Base type for expected project errors."""

    exit_code = 1


class DecisionBrainError(ORAgentError):
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequestError(DecisionBrainError):
    status_code = 400


class UserInputError(BadRequestError):
    """An input error that the user can correct."""


class ConfigurationError(DecisionBrainError):
    status_code = 500


class AgentRuntimeError(DecisionBrainError):
    """A Core Agent protocol or Runtime orchestration error."""

    status_code = 502


class LLMResponseError(AgentRuntimeError):
    """LLM provider returned a non-success HTTP response."""

    def __init__(self, provider_status: int, response_body: str):
        self.provider_status = provider_status
        self.response_body = response_body
        super().__init__(f"LLM HTTP {provider_status}: {response_body}")


class RunStorageError(ORAgentError):
    """A Run record cannot be created, read, or updated safely."""


class RunNotFoundError(RunStorageError):
    """The requested Run record does not exist."""


class InvalidRunPathError(RunStorageError):
    """A path inside a Run record is unsafe or invalid."""


class RunStateError(RunStorageError):
    """A Run transition or terminal-state update is invalid."""


class RunCapacityError(DecisionBrainError):
    """The service process has reached its active Run limit."""

    status_code = 429


class CorruptEventLogError(RunStorageError):
    """An event log contains corrupt, out-of-order, or mismatched records."""


class ArtifactConflictError(RunStorageError):
    """An artifact target exists and overwrite was not explicitly allowed."""


class ArtifactNotFoundError(RunStorageError):
    """Artifact metadata or content does not exist."""


class UnsupportedSchemaVersionError(RunStorageError):
    """Persisted data uses an unsupported schema version."""


class EventPublishingError(ORAgentError):
    """The event publisher cannot continue safely after a delivery failure."""
