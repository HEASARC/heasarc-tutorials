document.addEventListener("DOMContentLoaded", function() {
    var banner = document.createElement("div");
    banner.innerHTML = "The HEASARC tutorials resource is in <strong>BETA</strong> and may be subject to significant changes.";

    // CSS Styles for the banner
    banner.style.backgroundColor = "#EB9602";
    banner.style.color = "#000";
    banner.style.textAlign = "center";
    banner.style.padding = "1px";
    banner.style.fontWeight = "bold";
    banner.style.position = "fixed";
    banner.style.top = "0";
    banner.style.width = "100%";
    banner.style.zIndex = "9999";

    // Adjust body margin so content isn't hidden behind banner
    document.body.style.marginTop = "40px";

    // Insert at the very top of the body
    document.body.insertBefore(banner, document.body.firstChild);
});
