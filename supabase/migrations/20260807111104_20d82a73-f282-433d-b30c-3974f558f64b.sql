CREATE POLICY "Users read own uploads" ON storage.objects FOR SELECT TO authenticated
USING (bucket_id = 'chat-uploads' AND auth.uid()::text = (storage.foldername(name))[1]);
CREATE POLICY "Users insert own uploads" ON storage.objects FOR INSERT TO authenticated
WITH CHECK (bucket_id = 'chat-uploads' AND auth.uid()::text = (storage.foldername(name))[1]);
CREATE POLICY "Users delete own uploads" ON storage.objects FOR DELETE TO authenticated
USING (bucket_id = 'chat-uploads' AND auth.uid()::text = (storage.foldername(name))[1]);