const API = "/api";

const rowsElement = document.getElementById("rows");
const statusElement = document.getElementById("status");
const timeElement = document.getElementById("time");

const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");


// ===============================
// 상태 조회
// ===============================

async function updateStatus() {
    try {
        const response = await fetch(`${API}/status`);

        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }

        const data = await response.json();

        // 데이터 행 수
        rowsElement.innerText =
            Number(data.rows || 0).toLocaleString();

        // Collector 상태
        if (data.running) {
            statusElement.innerText = "● Running";
            statusElement.className = "green";

            startButton.disabled = true;
            stopButton.disabled = false;
        } else {
            statusElement.innerText = "● Stopped";
            statusElement.className = "red";

            startButton.disabled = false;
            stopButton.disabled = true;
        }

        // 마지막 상태 확인 시간
        timeElement.innerText =
            new Date().toLocaleString();

    } catch (error) {

        console.error(
            "Status update failed:",
            error
        );

        statusElement.innerText =
            "● Server Error";

        statusElement.className =
            "red";
    }
}


// ===============================
// Collector 시작
// ===============================

startButton.onclick = async function () {

    try {

        startButton.disabled = true;

        const response = await fetch(
            `${API}/collector/start`,
            {
                method: "POST"
            }
        );

        if (!response.ok) {
            throw new Error(
                `HTTP error: ${response.status}`
            );
        }

        const data =
            await response.json();

        console.log(
            "Collector start:",
            data
        );

        // 상태 바로 갱신
        await updateStatus();

    } catch (error) {

        console.error(
            "Collector start failed:",
            error
        );

        alert(
            "Collector 시작에 실패했습니다."
        );

        startButton.disabled = false;
    }
};


// ===============================
// Collector 종료
// ===============================

stopButton.onclick = async function () {

    try {

        stopButton.disabled = true;

        const response = await fetch(
            `${API}/collector/stop`,
            {
                method: "POST"
            }
        );

        if (!response.ok) {
            throw new Error(
                `HTTP error: ${response.status}`
            );
        }

        const data =
            await response.json();

        console.log(
            "Collector stop:",
            data
        );

        // 상태 바로 갱신
        await updateStatus();

    } catch (error) {

        console.error(
            "Collector stop failed:",
            error
        );

        alert(
            "Collector 종료에 실패했습니다."
        );

        stopButton.disabled = false;
    }
};


// ===============================
// 최초 상태 조회
// ===============================

updateStatus();


// ===============================
// 1초마다 상태 갱신
// ===============================

setInterval(
    updateStatus,
    1000
);