import { useState, useCallback, useEffect } from 'react';

export interface ChatSession {
  id: number;
  title: string;
  created_at: string;
}

export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  language: string;
  risk_level: 'safe' | 'caution' | 'banned' | 'unknown';
  timestamp: string;
}

export const useChat = () => {
  const [currentChatId, setCurrentChatId] = useState<number | null>(() => {
    const saved = localStorage.getItem('currentChatId');
    return saved ? parseInt(saved, 10) : null;
  });
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (currentChatId) {
      localStorage.setItem('currentChatId', currentChatId.toString());
    } else {
      localStorage.removeItem('currentChatId');
    }
  }, [currentChatId]);

  const fetchChats = useCallback(async () => {
    try {
      const res = await fetch('/chats');
      const data = await res.json();
      setChatSessions(data || []);
      return data;
    } catch (err) {
      console.error("Failed to fetch chats:", err);
      return [];
    }
  }, []);

  const loadChat = useCallback(async (chatId: number) => {
    setIsLoading(true);
    try {
      const res = await fetch(`/chats/${chatId}`);
      if (!res.ok) {
        localStorage.removeItem('currentChatId');
        throw new Error("Chat not found");
      }
      const data = await res.json();
      setMessages(data.messages || []);
      setCurrentChatId(chatId);
    } catch (err) {
      console.error("Failed to load chat:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createChat = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/chats', { method: 'POST' });
      const data = await res.json();
      await fetchChats();
      setCurrentChatId(data.chat_id);
      setMessages([]);
      return data.chat_id;
    } catch (err) {
      console.error("Failed to create chat:", err);
    } finally {
      setIsLoading(false);
    }
  }, [fetchChats]);

  const sendMessage = useCallback(async (query: string, chatIdOverride?: number) => {
    let targetChatId = chatIdOverride || currentChatId;
    
    if (!targetChatId) {
      targetChatId = await createChat();
      if (!targetChatId) return;
    }

    if (!query.trim()) return;

    setIsLoading(true);
    try {
      const res = await fetch(`/verify-text?chat_id=${targetChatId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      
      const data = await res.json();
      
      // Reload chat to get updated messages and title
      await loadChat(targetChatId);
      await fetchChats(); // Title might have changed
      
      return data;
    } catch (err) {
      console.error("Send message error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [currentChatId, createChat, loadChat, fetchChats]);

  const sendVoice = useCallback(async (audioBlob: Blob) => {
    let targetChatId = currentChatId;
    if (!targetChatId) {
      targetChatId = await createChat();
      if (!targetChatId) return;
    }

    setIsLoading(true);
    const formData = new FormData();
    // Derive correct file extension from the blob's MIME type so Whisper can parse the format
    const mimeType = audioBlob.type || 'audio/webm';
    const ext = mimeType.includes('mp4') ? 'mp4'
              : mimeType.includes('ogg') ? 'ogg'
              : 'webm';
    formData.append('audio', audioBlob, `recording.${ext}`);

    try {
      const res = await fetch(`/verify?chat_id=${targetChatId}`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await res.json();
      await loadChat(targetChatId);
      await fetchChats();
      return data;
    } catch (err) {
      console.error("Voice message error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [currentChatId, createChat, loadChat, fetchChats]);

  const deleteChat = useCallback(async (chatId: number) => {
    try {
      await fetch(`/chats/${chatId}`, { method: 'DELETE' });
      await fetchChats();
      if (currentChatId === chatId) {
        setMessages([]);
        setCurrentChatId(null);
      }
    } catch (err) {
      console.error("Delete chat error:", err);
    }
  }, [currentChatId, fetchChats]);

  useEffect(() => {
    const init = async () => {
      const sessions = await fetchChats();
      if (currentChatId) {
        loadChat(currentChatId);
      } else if (sessions && sessions.length > 0) {
        // Auto-load most recent chat if nothing in localStorage
        loadChat(sessions[0].id);
      }
    };
    init();
  }, []);

  return { 
    currentChatId, 
    chatSessions, 
    messages, 
    isLoading, 
    createChat, 
    loadChat, 
    sendMessage, 
    sendVoice, 
    deleteChat 
  };
};
