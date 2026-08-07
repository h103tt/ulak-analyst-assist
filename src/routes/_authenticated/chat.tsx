import { createFileRoute, Outlet } from "@tanstack/react-router";
import { ThreadSidebar } from "@/components/chat/ThreadSidebar";

export const Route = createFileRoute("/_authenticated/chat")({
  component: ChatLayout,
});

function ChatLayout() {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <ThreadSidebar />
      <main className="flex min-w-0 flex-1 flex-col">
        <Outlet />
      </main>
    </div>
  );
}
