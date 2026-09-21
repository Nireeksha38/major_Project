// Real-Time Dashboard Live Telemetry Poller
document.addEventListener("DOMContentLoaded", function () {
    const peopleElem = document.getElementById("val_people");
    const speedElem = document.getElementById("val_speed");
    const entropyElem = document.getElementById("val_entropy");
    const alertsElem = document.getElementById("val_alerts");

    function pollDashboardStats() {
        fetch("/api/latest-stats")
            .then(res => res.json())
            .then(data => {
                if (data) {
                    if (peopleElem) peopleElem.innerText = data.people_count || 0;
                    if (speedElem) speedElem.innerText = (data.relative_speed || 0).toFixed(2);
                    if (entropyElem) entropyElem.innerText = (data.motion_entropy || 0).toFixed(2);
                    if (alertsElem) alertsElem.innerText = data.active_alerts || 0;
                }
            })
            .catch(err => console.debug("Telemetry poll: " + err));
    }

    // Poll every 3 seconds
    setInterval(pollDashboardStats, 3000);
});
