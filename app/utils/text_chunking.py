import re
from typing import List

class TextChunking:
    """Utility class for text chunking operations"""
    
    @staticmethod
    def create_subtitle_chunks(text: str, chunk_size: int = 60, min_size: int = 40) -> List[str]:
        """Create subtitle chunks from text"""
        if not text:
            return []
        
        text = text.strip()
        if len(text) <= chunk_size:
            return [text]
        
        words = text.split()
        chunks = []
        current = words[0] if words else ""
        
        for word in words[1:]:
            if len(current) + 1 + len(word) <= chunk_size:
                current += " " + word
            else:
                if len(current) >= min_size:
                    chunks.append(current)
                    current = word
                else:
                    current += " " + word
        
        if current:
            chunks.append(current)
        
        return chunks
    
    @staticmethod
    def create_voice_chunks(text: str, chunk_size: int = 250, min_size: int = 220) -> List[str]:
        """Create voice generation chunks from text"""
        if not text:
            return []
        
        text = text.strip()
        if len(text) <= chunk_size:
            return [text]
        
        sentence_breaks = re.split(r'(?<=[.!?。！？؟؛…])\s+', text)
        chunks = []
        current = ""
        
        for sentence in sentence_breaks:
            if not sentence:
                continue
            if len(current) + len(sentence) <= chunk_size or len(current) < min_size:
                current = (current + " " + sentence).strip() if current else sentence
            else:
                chunks.append(current)
                current = sentence
        
        if current:
            chunks.append(current)
        
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                words = chunk.split()
                temp = words[0] if words else ""
                for word in words[1:]:
                    if len(temp) + 1 + len(word) <= chunk_size:
                        temp += " " + word
                    else:
                        final_chunks.append(temp)
                        temp = word
                if temp:
                    final_chunks.append(temp)
        
        return [c for c in final_chunks if c]
