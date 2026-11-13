export class ChatError extends Error {
  public readonly userMessage: string;
  public readonly status: number;

  constructor(userMessage: string, status = 500) {
    super(userMessage);
    this.name = "ChatError";
    this.userMessage = userMessage;
    this.status = status;
  }
}
