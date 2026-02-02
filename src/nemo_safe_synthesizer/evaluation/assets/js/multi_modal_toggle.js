function handleEvent(event) {
    const element = document.getElementById(event.currentTarget.dataset.for);
    if (element.hasAttribute("hidden")) {
        element.removeAttribute("hidden");
    }
    else {
        element.setAttribute("hidden", true);
    }
}

function toggleElement(event) {
    event.preventDefault();
    handleEvent(event);
}

function eraseMessage() {
    setTimeout(() => {
        document.getElementById("copied_message").innerHTML = ""
    }, 2000)
}

function copyModelOrWorkflowID() {
    // Get the model or workflow run id field
    var uid = document.getElementById("model_id") || document.getElementById("workflow_run_id");
    navigator.clipboard.writeText(uid.textContent.split(" "));

    document.getElementById("copied_message").innerHTML = "Copied"

    eraseMessage();
}

function toggleNav() {
    const navbar = document.getElementById("navbar");
    const nav = document.getElementById("fixed-nav");
    const navOverlayItems = document.getElementById("nav-overlay");
    const navBtn = document.getElementById("toggle-nav");
    const mainContent = document.querySelector("section.main");

    if (nav.style.visibility === "hidden") {
        nav.style.visibility = "visible";
        navOverlayItems.style.left = 0 + "px";
        navbar.style.backgroundColor = "white";
        navBtn.className = "opened-nav-button-bg";
        mainContent.style.paddingLeft = ""; //use CSS default when navbar is visible
    }
    else {
        nav.style.visibility = "hidden";
        navOverlayItems.style.left = -nav.offsetWidth + "px";
        navbar.style.backgroundColor = "transparent";
        navBtn.className = "closed-nav-button-bg";
        mainContent.style.paddingLeft = "24px";  
    }
}

const toggles = document.querySelectorAll("[data-toggle]");
toggles.forEach(toggle => { toggle.onclick = toggleElement });