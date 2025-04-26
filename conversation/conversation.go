package conversation

type ChatMessageRole string

const (
	ChatMessageRoleAssistant ChatMessageRole = "assistant"
	ChatMessageRoleSystem    ChatMessageRole = "system"
	ChatMessageRoleUser      ChatMessageRole = "user"
)

type ChatMessage struct {
	Content *string         `json:"content,omitempty"`
	Role    ChatMessageRole `json:"role"`
}

type Conversation struct {
	Messages     []ChatMessage
	SystemPrompt string
}

// Ptr returns a pointer to the given value.
func Ptr[T any](value T) *T {
	return &value
}

// AddMessage adds a message to the conversation.
func (c *Conversation) AddMessage(role ChatMessageRole, content string) {
	c.Messages = append(c.Messages, ChatMessage{
		Content: Ptr(content),
		Role:    role,
	})
}

// GetMessages returns the messages in the conversation.
func (c *Conversation) GetMessages() []ChatMessage {
	length := len(c.Messages)
	if c.SystemPrompt != "" {
		length++
	}

	messages := make([]ChatMessage, length)
	startIndex := 0

	if c.SystemPrompt != "" {
		messages[0] = ChatMessage{
			Content: Ptr(c.SystemPrompt),
			Role:    ChatMessageRoleSystem,
		}
		startIndex++
	}

	for i, message := range c.Messages {
		messages[startIndex+i] = message
	}

	return messages
}

// Reset removes messages from the conversation.
func (c *Conversation) Reset() {
	c.Messages = nil
}
