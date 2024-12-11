let audioEnabled = false;
const warningSound = new Audio('/static/warning.mp3');

document.getElementById('enableAudio').addEventListener('click', function() {
    audioEnabled = true;
    warningSound.play().then(() => {
        warningSound.pause();
        warningSound.currentTime = 0;
    });
    this.style.display = 'none';
});

// Sửa lại hàm xử lý WebSocket
const socket = new WebSocket(`ws://${window.location.host}/ws`);
socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.action === 'play_warning' && audioEnabled) {
        warningSound.play().then(() => {
            warningSound.currentTime = 0;
        }).catch(e => console.error('Error playing audio:', e));
    }
};