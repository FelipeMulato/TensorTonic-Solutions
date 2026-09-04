class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        saw_words = {}
        vocab = []
        words = []
        for text in texts:
            raw_text = text.split()
            for raw in raw_text:
                words.append(raw)
        for word in words:
            word = word.lower()
            if(word in saw_words):
                continue
            saw_words[word]=1
            vocab.append(word)
        vocab.sort()
        sz = len(vocab)
        self.vocab_size = sz+4

        raws = [self.pad_token,self.unk_token,self.bos_token,self.eos_token]
        id = 0
        for raw in raws:
            self.word_to_id[raw] = id
            self.id_to_word[id] = raw
            id+=1

        for word in vocab:
            self.word_to_id[word] = id
            self.id_to_word[id] = word
            id+=1
        
            

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        words = text.split()
        out = []
        for word in words:
            word = word.lower()
            if word in self.word_to_id:
                out.append(self.word_to_id[word])
            else:
                out.append(1)
        return out
    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        out = ""
        sz =len(ids)
        for i in range(0,sz):
            id = ids[i]
            if id in self.id_to_word:
                out+=self.id_to_word[id]
            else:
                out+="<UNK>"
            
            if i!=(sz-1):
                out+=" "

        return out