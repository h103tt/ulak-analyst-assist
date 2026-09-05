import { createFileRoute } from "@tanstack/react-router";
import { createClient } from "@supabase/supabase-js";
import type { UIMessage } from "ai";

type ChatRequestBody = {
  id?: unknown;
  messages?: unknown;
  context?: unknown;
  trigger?: unknown;
  messageId?: unknown;
  useQueryExpansion?: unknown;
};

const AGENT_URL = process.env["AGENT_BASE_URL"] ?? "http://127.0.0.1:8010";
const ATTACHMENT_BUCKET = "chat-uploads";
const SIGNED_URL_EXPIRY = 60 * 15; // 15 minutes, enough for the whole reply stream
// searchs for agent base url if it doesnt find any as default it goes to localhost

function messageText(message: UIMessage): string {
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("")
    .trim();
} // this turns the prompt into raw text so that the agent could understan it

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const token = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "");
        if (!token) return new Response("Unauthorized", { status: 401 });
        
        
        const supabaseUrl = process.env["SUPABASE_URL"];
        const publishableKey = process.env["SUPABASE_PUBLISHABLE_KEY"];
        if (!supabaseUrl || !publishableKey) {
          return new Response("Backend not configured", { status: 500 });
        }

        const supabase = createClient(supabaseUrl, publishableKey, {
          auth: { persistSession: false, autoRefreshToken: false },
          global: { headers: { Authorization: `Bearer ${token}`, apikey: publishableKey } },
        });
        const { data: userData, error: userError } = await supabase.auth.getUser(token);
        if (userError || !userData.user) return new Response("Unauthorized", { status: 401 });

        const body = (await request.json()) as ChatRequestBody;
        if (!Array.isArray(body.messages)) {
          return new Response("Messages are required", { status: 400 });
        }//checks whether the request made my someone authorized or not

        const contextText = typeof body.context === "string" ? body.context.slice(0, 60000) : "";
        const useQueryExpansion = body.useQueryExpansion === true;
        const agentMessages = (body.messages as UIMessage[])
          .map((message) => ({
            role: message.role,
            content: messageText(message),
          }))
          .filter((message) => message.content && message.role !== "system");

        if (agentMessages.length === 0) {
          return new Response("No usable messages", { status: 400 });
        }

        // Pass the thread's attachments to the agent so it can index them
        // into a session-scoped vector store. Signing happens here with the
        // user's token so the Python side only ever sees a short-lived URL.
        let files: Array<{ id: string; name: string; url: string }> = [];
        const threadId = typeof body.id === "string" ? body.id : "";
        if (threadId) {
          const { data: rows, error: listError } = await supabase
            .from("attachments")
            .select("id, file_name, storage_path")
            .eq("thread_id", threadId);
          if (!listError && rows && rows.length > 0) {
            const signed = await supabase.storage
              .from(ATTACHMENT_BUCKET)
              .createSignedUrls(rows.map((row) => row.storage_path), SIGNED_URL_EXPIRY);
            files = (signed.data ?? [])
              .filter((entry) => entry.signedUrl)
              .map((entry) => ({
                id: rows.find((row) => row.storage_path === entry.path)?.id ?? entry.path,
                name: rows.find((row) => row.storage_path === entry.path)?.file_name ?? entry.path,
                url: entry.signedUrl!,
              }));

            if (signed.error) {
              console.error("signed url creation failed", signed.error);
            }
          } else if (listError) {
            console.error("attachment list failed", listError);
          }
        }

        try {
          const agentResponse = await fetch(`${AGENT_URL}/chat`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              messages: agentMessages,
              context: contextText,
              thread_id: threadId || "default",
              files,
              use_query_expansion: useQueryExpansion,
            }),
            signal: request.signal,
          });

          if (!agentResponse.ok) {
            console.error(
              "analysis agent error",
              agentResponse.status,
              await agentResponse.text(),
            );
            return new Response("The analysis agent failed to generate a response", {
              status: 502,
            });
          }

          return new Response(agentResponse.body, {
            status: 200,
            headers: {
              "content-type": "text/event-stream; charset=utf-8",
              "cache-control": "no-cache",
              connection: "keep-alive",
            },
          });
        } catch (error) {
          console.error("analysis agent proxy failed", error);
          return new Response(
            "The analysis agent is not reachable. Is the Python agent server running?",
            { status: 503 },
          );
        }
      },
    },
  },
});
