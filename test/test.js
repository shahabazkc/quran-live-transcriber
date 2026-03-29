const { spawn } = require("child_process");

function transcribeAudio(audioPath) {
    return new Promise((resolve, reject) => {
        const process = spawn(
            "C:\\Users\\shaha\\OneDrive\\Desktop\\Shahabaz\\Projects\\whisper.cpp\\build\\bin\\whisper-cli.exe",
            [
                "-m", "C:\\Users\\shaha\\model-q4.bin",
                "-f", audioPath,
                "-t", "4" // threads
            ]
        );

        let output = "";
        let error = "";

        process.stdout.on("data", (data) => {
            output += data.toString();
        });

        process.stderr.on("data", (data) => {
            error += data.toString();
        });

        process.on("close", (code) => {
            if (code !== 0) {
                reject(error);
            } else {
                resolve(output);
            }
        });
    });
}

// Usage
(async () => {
    try {
        console.log("processing....");
        const startTime = Date.now();
        const result = await transcribeAudio(
            "C:\\Users\\shaha\\OneDrive\\Desktop\\Shahabaz\\Projects\\Whisper-finetuned-quran-dataset\\audio\\recording.wav"
        );
        const endTime = Date.now();
        console.log("Transcription:\n", result);
        console.log("Time taken:\n", endTime - startTime);
    } catch (err) {
        console.error("Error:", err);
    }
})();