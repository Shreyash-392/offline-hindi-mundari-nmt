package com.nmt.offline.translation

import java.io.File

/**
 * Offline SentencePiece BPE Tokenizer Runtime for Android
 * Loads vocabulary from tokenizer/hindi_mundari.vocab and performs subword BPE tokenization locally.
 */
class Tokenizer(vocabFile: File) {

    private val tokenToId = HashMap<String, Int>()
    private val idToToken = HashMap<Int, String>()

    val bosId = 1
    val eosId = 2
    val padId = 3
    val unkId = 0

    init {
        vocabFile.forEachLine { line ->
            val parts = line.split("\t")
            if (parts.isNotEmpty()) {
                val token = parts[0]
                val id = tokenToId.size
                tokenToId[token] = id
                idToToken[id] = token
            }
        }
    }

    /**
     * Encodes text into subword token IDs.
     */
    fun encode(text: String, maxLength: Int = 80): LongArray {
        val tokens = mutableListOf<Int>()
        val words = text.trim().split("\\s+".toRegex())

        for (word in words) {
            val subwords = bpeSegment(word)
            for (subword in subwords) {
                tokens.add(tokenToId[subword] ?: unkId)
            }
        }

        val result = LongArray(maxLength) { padId.toLong() }
        var idx = 0
        for (tokenId in tokens) {
            if (idx >= maxLength - 1) break
            result[idx++] = tokenId.toLong()
        }

        return result
    }

    /**
     * Decodes subword token IDs back into human readable text.
     */
    fun decode(tokenIds: List<Int>): String {
        val sb = StringBuilder()
        for (id in tokenIds) {
            if (id == bosId || id == padId) continue
            if (id == eosId) break

            val token = idToToken[id] ?: ""
            if (token.startsWith(" ")) {
                if (sb.isNotEmpty()) sb.append(" ")
                sb.append(token.substring(1))
            } else {
                sb.append(token)
            }
        }
        return sb.toString()
    }

    private fun bpeSegment(word: String): List<String> {
        val prepended = " $word"
        if (tokenToId.containsKey(prepended)) {
            return listOf(prepended)
        }
        // Subword greedy fallback
        val result = mutableListOf<String>()
        var start = 0
        while (start < prepended.length) {
            var end = prepended.length
            var matched = false
            while (end > start) {
                val sub = prepended.substring(start, end)
                if (tokenToId.containsKey(sub) || start > 0 && tokenToId.containsKey(sub)) {
                    result.add(sub)
                    start = end
                    matched = true
                    break
                }
                end--
            }
            if (!matched) {
                result.add(prepended.substring(start, start + 1))
                start++
            }
        }
        return result
    }
}
