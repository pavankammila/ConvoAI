
class ChatError(Exception):
    """An error that can be safely displayed to the user."""

    def __init__(self, user_message: str, status: int = 500):
        super().__init__(user_message)
        self.user_message = user_message
        self.status = status
