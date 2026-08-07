import { createFileRoute } from "@tanstack/react-router";
import { createClient } from "@supabase/supabase-js";
import { convertToModelMessages, streamText, type UIMessage } from "ai";
import { createLovableAiGatewayProvider } from "@/lib/ai-gateway.server";

type ChatRequestBody = {
  messages?: unknown;
  context?: unknown;
};

const SYSTEM_PROMPT = `You are the ULAK Quality Test Analyst, an AI assistant used internally by quality engineers at ULAK Haberleşme, a Turkish telecom equipment company.

Your expertise:
- Analysing software/hardware test reports, execution logs, defect records and coverage data
- Root-cause hypotheses, severity and priority assessment, regression risk
- Test strategy, test case design review, traceability to requirements
- Telecom domain context: 4G/5G RAN, core network, base stations, interoperability and field trials

How you work:
- Be precise, structured and evidence-driven. Cite the exact log lines, metrics or fields you relied on.
- Prefer clear markdown: short summary first, then findings as a table or list, then recommended actions.
- State assumptions explicitly and flag when data is insufficient rather than guessing.
- Reply in the user's language (Turkish or English).`;

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
        }

        const key = process.env["LOVABLE_API_KEY"];
        if (!key) return new Response("Missing AI credentials", { status: 500 });

        const contextText = typeof body.context === "string" ? body.context.slice(0, 60000) : "";
        const system = contextText
          ? `${SYSTEM_PROMPT}\n\nAttached files provided by the user in this conversation:\n${contextText}`
          : SYSTEM_PROMPT;

        const gateway = createLovableAiGatewayProvider(key);

        try {
          const result = streamText({
            model: gateway("google/gemini-3.5-flash"),
            system,
            messages: await convertToModelMessages(body.messages as UIMessage[]),
          });

          return result.toUIMessageStreamResponse({
            originalMessages: body.messages as UIMessage[],
          });
        } catch (error) {
          console.error("chat stream failed", error);
          return new Response("Failed to generate a response", { status: 500 });
        }
      },
    },
  },
});
