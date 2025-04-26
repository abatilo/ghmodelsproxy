package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/cli/go-gh/v2/pkg/api"
	"github.com/cli/go-gh/v2/pkg/auth"

	"github.com/abatilo/ghmodelsproxy/conversation"
	"github.com/abatilo/ghmodelsproxy/stream"
)

const (
	defaultInferenceURL = "https://models.github.ai/inference/chat/completions"
)

// AzureClientConfig represents configurable settings for the Azure client.
type AzureClientConfig struct {
	InferenceURL string
}

// ChatMessageRole represents the role of a chat message.
type ChatMessageRole string

const (
	// ChatMessageRoleUser represents a message from the user.
	ChatMessageRoleUser ChatMessageRole = "user"
)

// ChatMessage represents a message from a chat thread with a model.
type ChatMessage struct {
	Content *string         `json:"content,omitempty"`
	Role    ChatMessageRole `json:"role"`
}

type ChatCompletionOptions struct {
	Messages []ChatMessage `json:"messages"`
	Model    string        `json:"model"`
	Stream   bool          `json:"stream,omitempty"`
}

type chatChoiceDelta struct {
	Content *string `json:"content,omitempty"`
}

// ChatChoice represents a choice in a chat completion.
type ChatChoice struct {
	Delta *chatChoiceDelta `json:"delta,omitempty"`
}

// ChatCompletion represents a chat completion.
type ChatCompletion struct {
	Choices []ChatChoice `json:"choices"`
}

// ChatCompletionResponse represents a response to a chat completion request.
type ChatCompletionResponse struct {
	Reader stream.Reader[ChatCompletion]
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
		chatCompletionResponse.Reader = stream.NewEventReader[ChatCompletion](resp.Body)
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

func main() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [prompt]\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	var userPrompt string
	if flag.NArg() > 0 {
		userPrompt = flag.Arg(0)
	} else {
		userPrompt = "How do I get the length of a string in Python?"
	}

	token, _ := auth.TokenForHost("github.com")
	clientConfig := NewDefaultAzureClientConfig()
	client := NewAzureClient(http.DefaultClient, token, clientConfig)

	conv := conversation.Conversation{
		SystemPrompt: "You are a coding assistant",
		Messages: []conversation.ChatMessage{
			{
				Role:    conversation.ChatMessageRoleUser,
				Content: conversation.Ptr(userPrompt),
			},
		},
	}

	req := ChatCompletionOptions{
		Messages: []ChatMessage{}, // workaround for type, will copy below
		Model:    "OpenAI/gpt-4.1",
	}
	// Convert []conversation.ChatMessage to []ChatMessage
	req.Messages = make([]ChatMessage, len(conv.GetMessages()))
	for i, m := range conv.GetMessages() {
		req.Messages[i] = ChatMessage{
			Content: m.Content,
			Role:    ChatMessageRole(m.Role),
		}
	}

	resp, err := client.GetChatCompletionStream(context.TODO(), req)
	if err != nil {
		fmt.Println(err)
		return
	}
	reader := resp.Reader
	defer reader.Close()

	for {
		completion, err := reader.Read()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
		}

		for _, choice := range completion.Choices {
			if choice.Delta.Content != nil {
				fmt.Print(*choice.Delta.Content)
			}
		}
	}

	fmt.Println()
}
