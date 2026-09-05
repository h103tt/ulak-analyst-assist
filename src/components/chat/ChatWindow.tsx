import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileText, Loader2, Paperclip, Sparkles, X } from "lucide-react";
import { toast } from "sonner";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "@/components/ai-elements/prompt-input";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { supabase } from "@/integrations/supabase/client";
import {
  buildAttachmentContext,
  deleteAttachment,
  listAttachments,
  loadMessages,
  messageText,
  renameThread,
  saveMessage,
  touchThread,
  uploadAttachment,
  type Attachment,
} from "@/lib/chat-db";
//import logo from "@/public/ulak-logo-beyaz-2.png";

const SUGGESTIONS = [
  "Generate comprehensive functional test cases for the attached System Requirements Specification (SRS).",
  "Identify edge-case and negative test scenarios for the attached mission-critical communication requirements.",
  "Draft a test strategy and traceability matrix for the ULAK 5G RAN interoperability requirements.",
];

export function ChatWindow({ threadId, threadTitle }: { threadId: string; threadTitle: string }) {
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>(null);
  const savedIds = useRef<Set<string>>(new Set());

  useEffect(() => {
    let active = true;
    setInitialMessages(null);
    savedIds.current = new Set();
    loadMessages(threadId)
      .then((msgs) => {
        if (!active) return;
        msgs.forEach((m) => savedIds.current.add(m.id));
        setInitialMessages(msgs);
      })
      .catch(() => {
        if (active) {
          toast.error("Could not load this conversation");
          setInitialMessages([]);
        }
      });
    return () => {
      active = false;
    };
  }, [threadId]);

  // Show loader until messages are loaded, so useChat can initialize with the real history
  if (initialMessages === null) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }

  return <ChatWindowInner threadId={threadId} threadTitle={threadTitle} initialMessages={initialMessages} savedIds={savedIds} />;
}

function ChatWindowInner({ threadId, threadTitle, initialMessages, savedIds }: {
  threadId: string;
  threadTitle: string;
  initialMessages: UIMessage[];
  savedIds: React.MutableRefObject<Set<string>>;
}) {
  const queryClient = useQueryClient();
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [input, setInput] = useState("");
  const [thinkingMode, setThinkingMode] = useState(false);
  const thinkingModeRef = useRef(thinkingMode);
  useEffect(() => {
    thinkingModeRef.current = thinkingMode;
  }, [thinkingMode]);

  const attachments = useQuery({
    queryKey: ["attachments", threadId],
    queryFn: () => listAttachments(threadId),
  });

  const contextText = useMemo(
    () => buildAttachmentContext(attachments.data ?? []),
    [attachments.data],
  );
  const contextRef = useRef(contextText);
  useEffect(() => {
    contextRef.current = contextText;
  }, [contextText]);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        headers: async () => {
          const { data } = await supabase.auth.getSession();
          return data.session ? { Authorization: `Bearer ${data.session.access_token}` } : {};
        },
        body: () => ({ context: contextRef.current, useQueryExpansion: thinkingModeRef.current }),
      }),
    [],
  );

  const { messages, sendMessage, status, error } = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
    onError: (err) => toast.error(err.message || "The analyst could not respond"),
  });

  useEffect(() => {
    if (error) console.error(error);
  }, [error]);

  // Persist completed messages
  useEffect(() => {
    if (status === "streaming" || status === "submitted") return;
    const unsaved = messages.filter((m) => !savedIds.current.has(m.id) && messageText(m));
    if (unsaved.length === 0) return;
    unsaved.forEach((m) => savedIds.current.add(m.id));
    (async () => {
      for (const m of unsaved) {
        try {
          await saveMessage(threadId, m);
        } catch (e) {
          console.error("save message failed", e);
        }
      }
      try {
        await touchThread(threadId);
        await queryClient.invalidateQueries({ queryKey: ["threads"] });
      } catch (e) {
        console.error(e);
      }
    })();
  }, [messages, status, threadId, queryClient]);

  const focusInput = useCallback(() => {
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, []);

  useEffect(() => {
    focusInput();
  }, [threadId, focusInput]);
  useEffect(() => {
    if (status === "ready") focusInput();
  }, [status, focusInput]);

  const upload = useMutation({
    mutationFn: (file: File) => uploadAttachment(threadId, file),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["attachments", threadId] });
      toast.success("File attached to this conversation");
    },
    onError: () => toast.error("Upload failed"),
  });

  const removeFile = useMutation({
    mutationFn: (attachment: Attachment) => deleteAttachment(attachment),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["attachments", threadId] }),
    onError: () => toast.error("Could not remove that file"),
  });

  const submit = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || status === "streaming" || status === "submitted") return;
    setInput("");
    if (threadTitle === "New analysis" && messages.length === 0) {
      const title = trimmed.slice(0, 60);
      try {
        await renameThread(threadId, title);
        await queryClient.invalidateQueries({ queryKey: ["threads"] });
        await queryClient.invalidateQueries({ queryKey: ["thread", threadId] });
      } catch (e) {
        console.error(e);
      }
    }
    await sendMessage({ text: trimmed });
    focusInput();
  };

  const busy = status === "submitted" || status === "streaming";

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex h-[88px] shrink-0 items-center justify-between border-b border-border px-6">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-bold">{threadTitle}</h1>
          <p className="text-xs text-muted-foreground">
            {attachments.data?.length
              ? `${attachments.data.length} file(s) in context`
              : "No files attached"}
          </p>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple
          onChange={(e) => {
            const files = Array.from(e.target.files ?? []);
            files.forEach((f) => {
              if (f.size > 20_000_000) {
                toast.error(`${f.name} is larger than 20MB`);
                return;
              }
              upload.mutate(f);
            });
            e.target.value = "";
          }}
        />
        <Button
          variant="outline"
          className="rounded-full border-2 border-primary font-semibold"
          onClick={() => fileInputRef.current?.click()}
          disabled={upload.isPending}
        >
          {upload.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Paperclip className="h-4 w-4" />
          )}
          <span className="ml-2">Upload files</span>
        </Button>
      </header>

      {attachments.data && attachments.data.length > 0 && (
        <div className="flex flex-wrap gap-2 border-b border-border px-6 py-3">
          {attachments.data.map((a) => (
            <span
              key={a.id}
              className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground"
            >
              <FileText className="h-3.5 w-3.5 text-primary" />
              <span className="max-w-[220px] truncate">{a.file_name}</span>
              <button
                type="button"
                aria-label={`Remove ${a.file_name}`}
                onClick={() => removeFile.mutate(a)}
                className="transition-colors hover:text-destructive"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </span>
          ))}
        </div>
      )}

      <Conversation className="min-h-0 flex-1">
        <ConversationContent className="mx-auto w-full max-w-3xl px-6 py-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center py-16 text-center">
              <img src="/ulak-logo-beyaz-2.png" alt="ULAK agent mark" width={200} height={90} className="h-auto w-[180px]" />
              <h2 className="mt-6 text-2xl font-bold">How can I assist with your requirement analysis and test generation?</h2>
              <p className="mt-2 max-w-md text-sm text-muted-foreground">
                Attach System Requirement Specifications (SRS), interface control documents, or project specs to generate test cases, analyze traceability, or review requirement coverage.
              </p>
              <div className="mt-8 grid w-full gap-3">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => submit(s)}
                    className="panel px-5 py-4 text-left text-sm text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <Message key={message.id} from={message.role}>
                  <MessageContent>
                    <MessageResponse>{messageText(message)}</MessageResponse>
                  </MessageContent>
                </Message>
              ))}
              {status === "submitted" && <Shimmer>Analysing…</Shimmer>}
            </div>
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="shrink-0 border-t border-border px-6 py-4">
        <div className="mx-auto w-full max-w-3xl">
          <PromptInput
            onSubmit={(message, event) => {
              event.preventDefault();
              void submit(message.text ?? input);
            }}
          >
            <PromptInputTextarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about defects, coverage, logs or release risk…"
            />
            <PromptInputFooter className="justify-between">
              <PromptInputTools>
                <TooltipProvider>
                  <PromptInputButton
                    variant={thinkingMode ? "default" : "ghost"}
                    onClick={() => setThinkingMode((v) => !v)}
                    tooltip={{
                      content:
                        "Derinlemesine analiz: soruyu birden fazla açıdan arayıp daha kapsamlı yanıt üretir (daha yavaş, daha maliyetli).",
                    }}
                  >
                    <Sparkles className="size-4" />
                  </PromptInputButton>
                </TooltipProvider>
              </PromptInputTools>
              <PromptInputSubmit status={status} disabled={busy || input.trim().length === 0} />
            </PromptInputFooter>
          </PromptInput>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            Responses are AI-generated — verify findings against the source evidence.
          </p>
        </div>
      </div>
    </div>
  );
}
