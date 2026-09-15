const API = "/api";

// 상태 조회

function updateStatus(){

fetch(
API + "/status"
)

.then(
response =>
response.json()
)

.then(data=>{


document
.getElementById("rows")
.innerText =
data.rows.toLocaleString();



document
.getElementById("status")
.innerText =
data.running
?
"● Running"
:
"● Stopped";
document
.getElementById("time")
.innerText =
new Date()
.toLocaleString();
});

}

setInterval(
updateStatus,
1000
);

// 시작 버튼


document
.getElementById("start")
.onclick=function(){


fetch(
API+"/collector/start",
{
method:"POST"
}
);


};


// 종료 버튼
document
.getElementById("stop")
.onclick=function(){


fetch(
API+"/collector/stop",
{
method:"POST"
}
);


};