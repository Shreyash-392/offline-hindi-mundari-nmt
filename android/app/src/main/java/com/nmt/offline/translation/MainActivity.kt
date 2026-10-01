package com.nmt.offline.translation

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var translationEngine: TranslationEngine
    private lateinit var radioGroupDirection: RadioGroup
    private lateinit var editTextInput: EditText
    private lateinit var buttonTranslate: Button
    private lateinit var textViewOutput: TextView
    private lateinit var textViewStatus: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        radioGroupDirection = findViewById(R.id.radioGroupDirection)
        editTextInput = findViewById(R.id.editTextInput)
        buttonTranslate = findViewById(R.id.buttonTranslate)
        textViewOutput = findViewById(R.id.textViewOutput)
        textViewStatus = findViewById(R.id.textViewStatus)

        // Initialize local model files from assets
        initLocalTranslationEngine()

        buttonTranslate.setOnClickListener {
            performTranslation()
        }
    }

    private fun initLocalTranslationEngine() {
        try {
            textViewStatus.text = "Loading offline ONNX models..."

            val vocabFile = copyAssetToFile("hindi_mundari.vocab")
            val hiToMunModelFile = copyAssetToFile("transformer_hindi_mundari_int8.onnx")
            val munToHiModelFile = copyAssetToFile("transformer_mundari_hindi_int8.onnx")

            val tokenizer = Tokenizer(vocabFile)
            translationEngine = TranslationEngine(hiToMunModelFile, munToHiModelFile, tokenizer)

            textViewStatus.text = "Offline Translation Engine Ready (No Internet Required)"
        } catch (e: Exception) {
            textViewStatus.text = "Error initializing engine: ${e.message}"
        }
    }

    private fun performTranslation() {
        val inputText = editTextInput.text.toString().trim()
        if (inputText.isEmpty()) {
            textViewOutput.text = "Please enter text to translate."
            return
        }

        val direction = if (radioGroupDirection.checkedRadioButtonId == R.id.radioHiToMun) {
            TranslationDirection.HINDI_TO_MUNDARI
        } else {
            TranslationDirection.MUNDARI_TO_HINDI
        }

        val startTime = System.currentTimeMillis()
        val resultText = translationEngine.translate(inputText, direction)
        val elapsedMs = System.currentTimeMillis() - startTime

        textViewOutput.text = resultText
        textViewStatus.text = "Translation complete in ${elapsedMs} ms (100% Offline)"
    }

    private fun copyAssetToFile(assetName: String): File {
        val file = File(filesDir, assetName)
        if (!file.exists()) {
            assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::translationEngine.isInitialized) {
            translationEngine.close()
        }
    }
}
