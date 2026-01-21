(function() {
    // 1. Check if the user already unlocked the site in this session
    // If they have, we exit immediately and do nothing.
    if (sessionStorage.getItem('site_unlocked') === 'true') {
        return;
    }

    // 2. Define the CSS styles
    const styles = `
        #password-wall {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            display: flex; justify-content: center; align-items: center;
            z-index: 9999999;
            font-family: system-ui, -apple-system, sans-serif;
        }
        .password-content {
            background: white; padding: 40px; border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2); text-align: center;
            max-width: 400px; width: 90%;
        }
        .password-content input {
            width: 100%; padding: 12px; margin: 20px 0;
            border: 1px solid #ddd; border-radius: 8px; box-sizing: border-box;
        }
        .password-content button {
            width: 100%; padding: 12px; background: #EB9602;
            color: white; border: none; border-radius: 8px; cursor: pointer;
            font-weight: bold;
        }
    `;

    // 3. Define the HTML structure
    const html = `
        <div id="password-wall">
            <div class="password-content">
                <h2>Internal Review</h2>
                <p>HEASARC-tutorials is undergoing internal review. Please enter the password.</p>
                <input type="password" id="pass-input" placeholder="Password">
                <button id="pass-btn">Enter</button>
            </div>
        </div>
    `;

    const runGatekeeper = () => {
        // Inject styles
        const styleSheet = document.createElement("style");
        styleSheet.innerText = styles;
        document.head.appendChild(styleSheet);

        // Inject HTML
        document.body.insertAdjacentHTML('afterbegin', html);

        const btn = document.getElementById('pass-btn');
        const input = document.getElementById('pass-input');
        const correctPassword = "WeNamePhotons";

        const checkPass = () => {
            if (input.value === correctPassword) {
                // Save the state so they don't have to enter it again on this tab
                sessionStorage.setItem('site_unlocked', 'true');
                document.getElementById('password-wall').remove();
            } else {
                alert("Incorrect password.");
            }
        };

        btn.addEventListener('click', checkPass);
        input.addEventListener('keypress', (e) => { if (e.key === 'Enter') checkPass(); });
    };

    // 4. Ensure it runs as soon as the body is available
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", runGatekeeper);
    } else {
        runGatekeeper();
    }
})();
