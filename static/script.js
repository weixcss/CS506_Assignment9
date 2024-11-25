document.getElementById("experiment-form").addEventListener("submit", async function(event) {
    event.preventDefault(); // Prevent form submission

    // Collect input values from the form
    const activation = document.getElementById("activation").value;
    const lr = parseFloat(document.getElementById("lr").value);
    const stepNum = parseInt(document.getElementById("step_num").value);

    // Validation checks
    const validActivations = ["relu", "tanh", "sigmoid"];
    if (!validActivations.includes(activation)) {
        alert("Please choose a valid activation function: relu, tanh, or sigmoid.");
        return;
    }

    if (isNaN(lr) || lr <= 0) {
        alert("Please enter a valid positive number for the learning rate.");
        return;
    }

    if (isNaN(stepNum) || stepNum <= 0) {
        alert("Please enter a valid positive integer for the number of training steps.");
        return;
    }

    // Disable the submit button while the experiment runs
    const submitButton = event.target.querySelector("button[type='submit']");
    submitButton.disabled = true;
    submitButton.textContent = "Processing...";

    try {
        // Send the experiment parameters to the backend
        const response = await fetch("/run_experiment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ activation: activation, lr: lr, step_num: stepNum })
        });

        // Parse the JSON response
        const data = await response.json();

        if (response.ok) {
            // Show and update the results section
            const resultsDiv = document.getElementById("results");
            resultsDiv.style.display = "block";

            const resultImg = document.getElementById("result_gif");
            if (data.result_gif) {
                resultImg.src = `/${data.result_gif}`;
                resultImg.style.display = "block";
            } else {
                alert("Experiment completed, but the result GIF was not found.");
            }
        } else {
            // Handle errors returned from the backend
            alert(data.error || "An error occurred while running the experiment.");
        }
    } catch (error) {
        console.error("Error running experiment:", error);
        alert("A network error occurred. Please try again.");
    } finally {
        // Re-enable the submit button
        submitButton.disabled = false;
        submitButton.textContent = "Train and Visualize";
    }
});