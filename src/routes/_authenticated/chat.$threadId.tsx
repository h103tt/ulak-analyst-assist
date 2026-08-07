import { createFileRoute, useParams } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { ChatWindow } from "@/components/chat/ChatWindow";

export const Route = createFileRoute("/_authenticated/chat/$threadId")({
  component: ThreadPage,
});

function ThreadPage() {
  const { threadId } = useParams({ from: "/_authenticated/chat/$threadId" });

  const thread = useQuery({
    queryKey: ["thread", threadId],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("threads")
        .select("id, title")
        .eq("id", threadId)
        .maybeSingle();
      if (error) throw error;
      return data;
    },
  });

  if (thread.isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  if (!thread.data) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
        <h1 className="text-xl font-bold">Conversation not found</h1>
        <p className="text-sm text-muted-foreground">
          It may have been deleted, or it belongs to another account.
        </p>
      </div>
    );
  }

  return <ChatWindow key={threadId} threadId={threadId} threadTitle={thread.data.title} />;
}
