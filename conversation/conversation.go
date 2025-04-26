package conversation

import "github.com/cli/go-gh/v2/pkg/api"

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
	messages     []ChatMessage
	systemPrompt string
}

// Ptr returns a pointer to the given value.
func Ptr[T any](value T) *T {
	return &value
}

// AddMessage adds a message to the conversation.
func (c *Conversation) AddMessage(role ChatMessageRole, content string) {
	c.messages = append(c.messages, ChatMessage{
		Content: Ptr(content),
		Role:    role,
	})
}

// GetMessages returns the messages in the conversation.
func GetMessages(c *Conversation) []ChatMessage {
	length := len(c.messages)
	if c.systemPrompt != "" {
		length++
	}

	messages := make([]ChatMessage, length)
	startIndex := 0

	if c.systemPrompt != "" {
		messages[0] = ChatMessage{
			Content: Ptr(c.systemPrompt),
			Role:    ChatMessageRoleSystem,
		}
		startIndex++
	}

	for i, message := range c.messages {
		messages[startIndex+i] = message
	}

	return messages
}

// Reset removes messages from the conversation.
func (c *Conversation) Reset() {
	c.messages = nil
}
