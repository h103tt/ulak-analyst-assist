import { supabase } from "@/integrations/supabase/client";
import type { UIMessage } from "ai";

export type Thread = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type Attachment = {
  id: string;
  file_name: string;
  mime_type: string | null;
  size_bytes: number | null;
  storage_path: string;
  extracted_text: string | null;
  created_at: string;
};

export const ATTACHMENT_BUCKET = "chat-uploads";
const MAX_TEXT_CHARS = 20000;

export async function requireUserId(): Promise<string> {
  const { data, error } = await supabase.auth.getUser();
  if (error || !data.user) throw new Error("Not signed in");
  return data.user.id;
}

export async function listThreads(): Promise<Thread[]> {
  const { data, error } = await supabase
    .from("threads")
    .select("id, title, created_at, updated_at")
    .order("updated_at", { ascending: false });
  if (error) throw error;
  return data ?? [];
}

export async function createThread(title = "New analysis"): Promise<Thread> {
  const userId = await requireUserId();
  const { data, error } = await supabase
    .from("threads")
    .insert({ user_id: userId, title })
    .select("id, title, created_at, updated_at")
    .single();
  if (error) throw error;
  return data;
}

export async function renameThread(id: string, title: string) {
  const { error } = await supabase.from("threads").update({ title }).eq("id", id);
  if (error) throw error;
}

export async function touchThread(id: string) {
  const { error } = await supabase
    .from("threads")
    .update({ updated_at: new Date().toISOString() })
    .eq("id", id);
  if (error) throw error;
}

export async function deleteThread(id: string) {
  const { error } = await supabase.from("threads").delete().eq("id", id);
  if (error) throw error;
}

export async function loadMessages(threadId: string): Promise<UIMessage[]> {
  const { data, error } = await supabase
    .from("messages")
    .select("id, role, content, created_at")
    .eq("thread_id", threadId)
    .order("created_at", { ascending: true });
  if (error) throw error;
  return (data ?? []).map((row) => ({
    id: row.id,
    role: row.role as UIMessage["role"],
    parts: [{ type: "text" as const, text: row.content }],
  }));
}

export function messageText(message: UIMessage): string {
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("")
    .trim();
}

export async function saveMessage(threadId: string, message: UIMessage) {
  const userId = await requireUserId();
  const { error } = await supabase.from("messages").insert({
    thread_id: threadId,
    user_id: userId,
    role: message.role,
    content: messageText(message),
    client_id: message.id,
  });
  if (error) throw error;
}

export async function listAttachments(threadId: string): Promise<Attachment[]> {
  const { data, error } = await supabase
    .from("attachments")
    .select("id, file_name, mime_type, size_bytes, storage_path, extracted_text, created_at")
    .eq("thread_id", threadId)
    .order("created_at", { ascending: true });
  if (error) throw error;
  return data ?? [];
}

function isTextLike(file: File) {
  if (file.type.startsWith("text/")) return true;
  return /\.(txt|log|csv|json|xml|md|yml|yaml|ini|conf|html|sql)$/i.test(file.name);
}

export async function uploadAttachment(threadId: string, file: File): Promise<Attachment> {
  const userId = await requireUserId();
  const safeName = file.name.replace(/[^\w.\-]+/g, "_").slice(-120);
  const path = `${userId}/${threadId}/${crypto.randomUUID()}-${safeName}`;

  const { error: uploadError } = await supabase.storage
    .from(ATTACHMENT_BUCKET)
    .upload(path, file, { contentType: file.type || "application/octet-stream" });
  if (uploadError) throw uploadError;

  let extracted: string | null = null;
  if (isTextLike(file) && file.size < 5_000_000) {
    extracted = (await file.text()).slice(0, MAX_TEXT_CHARS);
  }

  const { data, error } = await supabase
    .from("attachments")
    .insert({
      thread_id: threadId,
      user_id: userId,
      file_name: file.name,
      mime_type: file.type || null,
      size_bytes: file.size,
      storage_path: path,
      extracted_text: extracted,
    })
    .select("id, file_name, mime_type, size_bytes, storage_path, extracted_text, created_at")
    .single();
  if (error) throw error;
  return data;
}

export async function deleteAttachment(attachment: Attachment) {
  await supabase.storage.from(ATTACHMENT_BUCKET).remove([attachment.storage_path]);
  const { error } = await supabase.from("attachments").delete().eq("id", attachment.id);
  if (error) throw error;
}

export function buildAttachmentContext(attachments: Attachment[]): string {
  if (attachments.length === 0) return "";
  return attachments
    .map((a) => {
      const header = `FILE: ${a.file_name} (${a.mime_type ?? "unknown type"}, ${
        a.size_bytes ?? 0
      } bytes)`;
      return a.extracted_text
        ? `${header}\n---\n${a.extracted_text}\n---`
        : `${header}\n(binary file — contents not extracted)`;
    })
    .join("\n\n");
}
