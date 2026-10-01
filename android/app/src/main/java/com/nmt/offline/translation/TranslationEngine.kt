package com.nmt.offline.translation

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.LongBuffer

enum class TranslationDirection {
    HINDI_TO_MUNDARI,
    MUNDARI_TO_HINDI
}

/**
 * Local Offline ONNX Machine Translation Engine for Android
 * Executes dynamic INT8 quantized Transformer model entirely on mobile CPU without internet connection.
 */
class TranslationEngine(
    private val hiToMunModelFile: File,
    private val munToHiModelFile: File,
    private val tokenizer: Tokenizer
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var hiToMunSession: OrtSession? = null
    private var munToHiSession: OrtSession? = null

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(2)
        hiToMunSession = env.createSession(hiToMunModelFile.absolutePath, opts)
        munToHiSession = env.createSession(munToHiModelFile.absolutePath, opts)
    }

    /**
     * Translates input text offline using local ONNX INT8 model runtime.
     */
    fun translate(
        text: String,
        direction: TranslationDirection,
        maxLength: Int = 80
    ): String {
        val session = if (direction == TranslationDirection.HINDI_TO_MUNDARI) hiToMunSession else munToHiSession
            ?: return "Error: Model session not initialized"

        val srcIds = tokenizer.encode(text, maxLength)
        val srcPaddingMask = BooleanArray(maxLength) { srcIds[it] == tokenizer.padId.toLong() }

        val generated = mutableListOf<Int>(tokenizer.bosId)

        for (step in 0 until maxLength - 1) {
            val tgtPadded = LongArray(maxLength) { tokenizer.padId.toLong() }
            for (i in generated.indices) {
                tgtPadded[i] = generated[i].toLong()
            }
            val tgtPaddingMask = BooleanArray(maxLength) { tgtPadded[it] == tokenizer.padId.toLong() }

            // Prepare 2D causal mask
            val tgtMaskArray = BooleanArray(maxLength * maxLength) { idx ->
                val row = idx / maxLength
                val col = idx % maxLength
                col > row
            }

            // Create ONNX Tensors
            val srcTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(srcIds), longArrayOf(1, maxLength.toLong()))
            val tgtTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tgtPadded), longArrayOf(1, maxLength.toLong()))
            val srcMaskTensor = OnnxTensor.createTensor(env, srcPaddingMask)
            val tgtMaskTensor = OnnxTensor.createTensor(env, tgtPaddingMask)

            val inputs = mapOf(
                "src" to srcTensor,
                "tgt" to tgtTensor,
                "src_padding_mask" to srcMaskTensor,
                "tgt_padding_mask" to tgtMaskTensor
            )

            val result = session.run(inputs)
            val logits = result[0].value as Array<Array<FloatArray>> // shape: (1, 80, vocab_size)

            val currentPos = generated.size - 1
            val posLogits = logits[0][currentPos]

            var bestToken = 0
            var maxLogit = Float.NEGATIVE_INFINITY
            for (v in posLogits.indices) {
                if (posLogits[v] > maxLogit) {
                    maxLogit = posLogits[v]
                    bestToken = v
                }
            }

            generated.add(bestToken)
            if (bestToken == tokenizer.eosId) break
        }

        return tokenizer.decode(generated)
    }

    fun close() {
        hiToMunSession?.close()
        munToHiSession?.close()
        env.close()
    }
}
