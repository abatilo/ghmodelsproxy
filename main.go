package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/cli/go-gh/v2/pkg/api"
	"github.com/cli/go-gh/v2/pkg/auth"
)

const (
	defaultInferenceURL = "https://models.github.ai/inference/chat/completions"
)

// AzureClientConfig represents configurable settings for the Azure client.
type AzureClientConfig struct {
	InferenceURL     string
	AzureAiStudioURL string
	ModelsURL        string
}

// ChatMessageRole represents the role of a chat message.
type ChatMessageRole string

const (
	// ChatMessageRoleAssistant represents a message from the model.
	ChatMessageRoleAssistant ChatMessageRole = "assistant"
	// ChatMessageRoleSystem represents a system message.
	ChatMessageRoleSystem ChatMessageRole = "system"
	// ChatMessageRoleUser represents a message from the user.
	ChatMessageRoleUser ChatMessageRole = "user"
)

// ChatMessage represents a message from a chat thread with a model.
type ChatMessage struct {
	Content *string         `json:"content,omitempty"`
	Role    ChatMessageRole `json:"role"`
}

// ChatCompletionOptions represents available options for a chat completion request.
type ChatCompletionOptions struct {
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Messages    []ChatMessage `json:"messages"`
	Model       string        `json:"model"`
	Stream      bool          `json:"stream,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
}

// ChatChoiceMessage is a message from a choice in a chat conversation.
type ChatChoiceMessage struct {
	Content *string `json:"content,omitempty"`
	Role    *string `json:"role,omitempty"`
}

type chatChoiceDelta struct {
	Content *string `json:"content,omitempty"`
	Role    *string `json:"role,omitempty"`
}

// ChatChoice represents a choice in a chat completion.
type ChatChoice struct {
	Delta        *chatChoiceDelta   `json:"delta,omitempty"`
	FinishReason string             `json:"finish_reason"`
	Index        int32              `json:"index"`
	Message      *ChatChoiceMessage `json:"message,omitempty"`
}

// ChatCompletion represents a chat completion.
type ChatCompletion struct {
	Choices []ChatChoice `json:"choices"`
}

// ChatCompletionResponse represents a response to a chat completion request.
type ChatCompletionResponse struct {
	Reader Reader[ChatCompletion]
}

type modelCatalogSearchResponse struct {
	Summaries []modelCatalogSearchSummary `json:"summaries"`
}

type modelCatalogSearchSummary struct {
	AssetID        string      `json:"assetId"`
	DisplayName    string      `json:"displayName"`
	InferenceTasks []string    `json:"inferenceTasks"`
	Name           string      `json:"name"`
	Popularity     json.Number `json:"popularity"`
	Publisher      string      `json:"publisher"`
	RegistryName   string      `json:"registryName"`
	Version        string      `json:"version"`
	Summary        string      `json:"summary"`
}

type modelCatalogTextLimits struct {
	MaxOutputTokens    int `json:"maxOutputTokens"`
	InputContextWindow int `json:"inputContextWindow"`
}

type modelCatalogLimits struct {
	SupportedLanguages        []string                `json:"supportedLanguages"`
	TextLimits                *modelCatalogTextLimits `json:"textLimits"`
	SupportedInputModalities  []string                `json:"supportedInputModalities"`
	SupportedOutputModalities []string                `json:"supportedOutputModalities"`
}

type modelCatalogPlaygroundLimits struct {
	RateLimitTier string `json:"rateLimitTier"`
}

type modelCatalogDetailsResponse struct {
	AssetID            string                        `json:"assetId"`
	Name               string                        `json:"name"`
	DisplayName        string                        `json:"displayName"`
	Publisher          string                        `json:"publisher"`
	Version            string                        `json:"version"`
	RegistryName       string                        `json:"registryName"`
	Evaluation         string                        `json:"evaluation"`
	Summary            string                        `json:"summary"`
	Description        string                        `json:"description"`
	License            string                        `json:"license"`
	LicenseDescription string                        `json:"licenseDescription"`
	Notes              string                        `json:"notes"`
	Keywords           []string                      `json:"keywords"`
	InferenceTasks     []string                      `json:"inferenceTasks"`
	FineTuningTasks    []string                      `json:"fineTuningTasks"`
	Labels             []string                      `json:"labels"`
	TradeRestricted    bool                          `json:"tradeRestricted"`
	CreatedTime        string                        `json:"createdTime"`
	PlaygroundLimits   *modelCatalogPlaygroundLimits `json:"playgroundLimits"`
	ModelLimits        *modelCatalogLimits           `json:"modelLimits"`
}

// Client represents a client for interacting with an API about models.
type Client interface {
	// GetChatCompletionStream returns a stream of chat completions using the given options.
	GetChatCompletionStream(context.Context, ChatCompletionOptions) (*ChatCompletionResponse, error)
}

// NewDefaultAzureClientConfig returns a new AzureClientConfig with default values for API URLs.
func NewDefaultAzureClientConfig() *AzureClientConfig {
	return &AzureClientConfig{
		InferenceURL: defaultInferenceURL,
	}
}

// AzureClient provides a client for interacting with the Azure models API.
type AzureClient struct {
	client *http.Client
	token  string
	cfg    *AzureClientConfig
}

// NewDefaultAzureClient returns a new Azure client using the given auth token using default API URLs.
func NewDefaultAzureClient(authToken string) (*AzureClient, error) {
	httpClient, err := api.DefaultHTTPClient()
	if err != nil {
		return nil, err
	}
	cfg := NewDefaultAzureClientConfig()
	return &AzureClient{client: httpClient, token: authToken, cfg: cfg}, nil
}

// NewAzureClient returns a new Azure client using the given HTTP client, configuration, and auth token.
func NewAzureClient(httpClient *http.Client, authToken string, cfg *AzureClientConfig) *AzureClient {
	return &AzureClient{client: httpClient, token: authToken, cfg: cfg}
}

// GetChatCompletionStream returns a stream of chat completions using the given options.
func (c *AzureClient) GetChatCompletionStream(ctx context.Context, req ChatCompletionOptions) (*ChatCompletionResponse, error) {
	// Check for o1 models, which don't support streaming
	if req.Model == "o1-mini" || req.Model == "o1-preview" || req.Model == "o1" {
		req.Stream = false
	} else {
		req.Stream = true
	}

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	body := bytes.NewReader(bodyBytes)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.InferenceURL, body)
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Authorization", "Bearer "+c.token)
	httpReq.Header.Set("Content-Type", "application/json")

	// Azure would like us to send specific user agents to help distinguish
	// traffic from known sources and other web requests
	httpReq.Header.Set("x-ms-useragent", "github-cli-models")
	httpReq.Header.Set("x-ms-user-agent", "github-cli-models") // send both to accommodate various Azure consumers

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		// If we aren't going to return an SSE stream, then ensure the response body is closed.
		defer resp.Body.Close()
		return nil, c.handleHTTPError(resp)
	}

	var chatCompletionResponse ChatCompletionResponse

	if req.Stream {
		// Handle streamed response
		chatCompletionResponse.Reader = NewEventReader[ChatCompletion](resp.Body)
	}

	return &chatCompletionResponse, nil
}

func (c *AzureClient) handleHTTPError(resp *http.Response) error {
	sb := strings.Builder{}
	var err error

	switch resp.StatusCode {
	case http.StatusUnauthorized:
		_, err = sb.WriteString("unauthorized")
		if err != nil {
			return err
		}

	case http.StatusBadRequest:
		_, err = sb.WriteString("bad request")
		if err != nil {
			return err
		}

	default:
		_, err = sb.WriteString("unexpected response from the server: " + resp.Status)
		if err != nil {
			return err
		}
	}

	body, _ := io.ReadAll(resp.Body)
	if len(body) > 0 {
		_, err = sb.WriteString("\n")
		if err != nil {
			return err
		}

		_, err = sb.Write(body)
		if err != nil {
			return err
		}

		_, err = sb.WriteString("\n")
		if err != nil {
			return err
		}
	}

	return errors.New(sb.String())
}

// Reader is an interface for reading events from an SSE stream.
type Reader[T any] interface {
	// Read reads the next event from the stream.
	// Returns io.EOF when there are no further events.
	Read() (T, error)
	// Close closes the Reader and any applicable inner stream state.
	Close() error
}

// EventReader streams events dynamically from an OpenAI endpoint.
type EventReader[T any] struct {
	reader  io.ReadCloser // Required for Closing
	scanner *bufio.Scanner
}

// NewEventReader creates an EventReader that provides access to messages of
// type T from r.
func NewEventReader[T any](r io.ReadCloser) *EventReader[T] {
	return &EventReader[T]{reader: r, scanner: bufio.NewScanner(r)}
}

// Read reads the next event from the stream.
// Returns io.EOF when there are no further events.
func (er *EventReader[T]) Read() (T, error) {
	// https://html.spec.whatwg.org/multipage/server-sent-events.html
	for er.scanner.Scan() { // Scan while no error
		line := er.scanner.Text() // Get the line & interpret the event stream:

		if line == "" || line[0] == ':' { // If the line is blank or is a comment, skip it
			continue
		}

		if strings.Contains(line, ":") { // If the line contains a U+003A COLON character (:), process the field
			tokens := strings.SplitN(line, ":", 2)
			tokens[0], tokens[1] = strings.TrimSpace(tokens[0]), strings.TrimSpace(tokens[1])
			var data T
			switch tokens[0] {
			case "data": // return the deserialized JSON object
				if tokens[1] == "[DONE]" { // If data is [DONE], end of stream was reached
					return data, io.EOF
				}
				err := json.Unmarshal([]byte(tokens[1]), &data)
				return data, err
			default: // Any other event type is an unexpected
				return data, errors.New("unexpected event type: " + tokens[0])
			}
			// Unreachable
		}
	}

	scannerErr := er.scanner.Err()

	if scannerErr == nil {
		return *new(T), errors.New("incomplete stream")
	}

	return *new(T), scannerErr
}

// Close closes the EventReader and any applicable inner stream state.
func (er *EventReader[T]) Close() error {
	return er.reader.Close()
}

// Conversation represents a conversation between the user and the model.
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
func (c *Conversation) GetMessages() []ChatMessage {
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

func main() {
	token, _ := auth.TokenForHost("github.com")
	clientConfig := NewDefaultAzureClientConfig()
	client := NewAzureClient(http.DefaultClient, token, clientConfig)

	conversation := Conversation{
		systemPrompt: "You are a coding assistant",
		messages: []ChatMessage{
			{
				Role:    ChatMessageRoleUser,
				Content: Ptr("How do I get the length of a string in Python?"),
			},
		},
	}

	req := ChatCompletionOptions{
		Messages: conversation.GetMessages(),
		Model:    "OpenAI/gpt-4.1",
	}

	resp, err := client.GetChatCompletionStream(context.TODO(), req)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	reader := resp.Reader
	defer reader.Close()

	messageBuilder := strings.Builder{}

	for {
		completion, err := reader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			// return err
		}

		for _, choice := range completion.Choices {
			if choice.Delta.Content != nil {
				messageBuilder.WriteString(*choice.Delta.Content)
			}
			// fmt.Printf("%#v\n", *choice.Delta.Content)
			// err = cmdHandler.handleCompletionChoice(choice, messageBuilder)
			// if err != nil {
			// 	return err
			// }
		}
	}

	fmt.Println(messageBuilder.String())
}
