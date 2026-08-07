import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LogOut, MessageSquarePlus, Trash2, PanelLeftClose, PanelLeft } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { createThread, deleteThread, listThreads } from "@/lib/chat-db";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import logo from "@/assets/ulak-logo-beyaz.png.asset.json";

export function ThreadSidebar() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [collapsed, setCollapsed] = useState(false);
  const [profileName, setProfileName] = useState<string>("");
  const params = useParams({ strict: false }) as { threadId?: string };

  useEffect(() => {
    let active = true;
    supabase.auth.getUser().then(async ({ data }) => {
      if (!data.user) return;
      const { data: profile } = await supabase
        .from("profiles")
        .select("display_name")
        .eq("id", data.user.id)
        .maybeSingle();
      if (active) setProfileName(profile?.display_name || data.user.email || "Engineer");
    });
    return () => {
      active = false;
    };
  }, []);

  const threads = useQuery({ queryKey: ["threads"], queryFn: listThreads });

  const newThread = useMutation({
    mutationFn: () => createThread(),
    onSuccess: async (thread) => {
      await queryClient.invalidateQueries({ queryKey: ["threads"] });
      navigate({ to: "/chat/$threadId", params: { threadId: thread.id } });
    },
    onError: () => toast.error("Could not start a new conversation"),
  });

  const removeThread = useMutation({
    mutationFn: (id: string) => deleteThread(id),
    onSuccess: async (_data, id) => {
      await queryClient.invalidateQueries({ queryKey: ["threads"] });
      if (params.threadId === id) navigate({ to: "/chat" });
    },
    onError: () => toast.error("Could not delete that conversation"),
  });

  const signOut = async () => {
    await queryClient.cancelQueries();
    queryClient.clear();
    await supabase.auth.signOut();
    navigate({ to: "/auth", replace: true });
  };

  return (
    <aside
      className={cn(
        "flex h-full shrink-0 flex-col border-r border-sidebar-border bg-sidebar transition-[width] duration-200",
        collapsed ? "w-[72px]" : "w-[300px]",
      )}
    >
      <div className="flex h-[88px] items-center gap-3 border-b border-sidebar-border px-4">
        <Link to="/chat" className="flex min-w-0 items-center gap-3">
          <img src={logo.url} alt="ULAK agent mark" width={36} height={36} className="h-9 w-9 shrink-0" />
          {!collapsed && (
            <div className="min-w-0 leading-tight">
              <p className="truncate text-base font-extrabold tracking-tight">ULAK</p>
              <p className="truncate text-xs text-muted-foreground">Quality Test Analyst</p>
            </div>
          )}
        </Link>
        <Button
          variant="ghost"
          size="icon-sm"
          className="ml-auto text-muted-foreground"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </Button>
      </div>

      <div className="p-3">
        <Button
          onClick={() => newThread.mutate()}
          disabled={newThread.isPending}
          className={cn("w-full rounded-full font-bold", collapsed && "px-0")}
        >
          <MessageSquarePlus className="h-4 w-4" />
          {!collapsed && <span className="ml-2">New analysis</span>}
        </Button>
      </div>

      <nav className="flex-1 space-y-1 overflow-y-auto px-3 pb-3">
        {!collapsed && (
          <p className="px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
            History
          </p>
        )}
        {threads.data?.map((thread) => {
          const active = params.threadId === thread.id;
          return (
            <div
              key={thread.id}
              className={cn(
                "group flex items-center gap-1 rounded-lg pr-1 transition-colors",
                active ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
              )}
            >
              <Link
                to="/chat/$threadId"
                params={{ threadId: thread.id }}
                className={cn(
                  "min-w-0 flex-1 truncate px-3 py-2.5 text-sm transition-colors",
                  active ? "font-semibold text-primary" : "text-sidebar-foreground",
                )}
                title={thread.title}
              >
                {collapsed ? thread.title.slice(0, 1).toUpperCase() : thread.title}
              </Link>
              {!collapsed && (
                <button
                  type="button"
                  aria-label={`Delete ${thread.title}`}
                  onClick={() => removeThread.mutate(thread.id)}
                  className="rounded-md p-1.5 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
            </div>
          );
        })}
        {threads.data?.length === 0 && !collapsed && (
          <p className="px-2 text-sm text-muted-foreground">No conversations yet.</p>
        )}
      </nav>

      <div className="border-t border-sidebar-border p-3">
        {!collapsed && (
          <p className="truncate px-2 pb-2 text-xs text-muted-foreground">{profileName}</p>
        )}
        <Button
          variant="ghost"
          onClick={signOut}
          className={cn("w-full justify-start text-muted-foreground", collapsed && "justify-center px-0")}
        >
          <LogOut className="h-4 w-4" />
          {!collapsed && <span className="ml-2">Sign out</span>}
        </Button>
      </div>
    </aside>
  );
}
