document.addEventListener("DOMContentLoaded", function () {
  // Get all accordion buttons
  const accordions = document.querySelectorAll(".accordion");

  // Add click event listener to each accordion
  accordions.forEach((accordion) => {
    accordion.addEventListener("click", function () {
      const panel = this.nextElementSibling; // The panel is the next sibling of the button
      const chevron = this.querySelector("i"); // Get the chevron icon inside the accordion

      // If the panel is already open, collapse it
      if (panel.style.maxHeight && panel.style.maxHeight !== "0px") {
        panel.style.maxHeight = "0"; // Collapse the panel by setting max-height to 0
        panel.style.padding = "0 18px"; // Reset padding when collapsed
        // Rotate the chevron back to its original state
        chevron.style.transform = "rotate(0deg)";
      } else {
        // Close all panels first
        document.querySelectorAll(".accordion--panel").forEach((p) => {
          p.style.maxHeight = "0"; // Collapse all other panels
          p.style.padding = "0 18px"; // Reset padding for all panels
          // Reset the icons to show collapsed state for all panels
          const icon = p.previousElementSibling.querySelector("i");
          icon.style.transform = "rotate(0deg)"; // Reset rotation for all chevrons
        });

        // Open the current panel by setting max-height to the scrollHeight of the content
        panel.style.maxHeight = panel.scrollHeight + "px"; // Open the panel by setting max-height to scrollHeight
        panel.style.padding = "18px"; // Add padding when opened
        // Rotate the chevron 180 degrees
        chevron.style.transform = "rotate(-180deg)";
      }
    });
  });
});

function shutdownNow(event) {
  event.preventDefault();

  // Show the confirmation modal
  const confirmModal = document.getElementById("confirmModal");
  confirmModal.style.display = "flex"; // Display the modal

  // When the user clicks "Yes"
  document.getElementById("confirmYes").onclick = function () {
    // Hide the confirmation modal
    confirmModal.style.display = "none";

    // Show the loader and dimmed background
    const shutdownContainer = document.querySelector(".shutdown");
    const loaderWrapper = document.querySelector(".shutdownLoader__wrapper");

    if (shutdownContainer && loaderWrapper) {
      shutdownContainer.style.display = "flex"; // Show the container
      loaderWrapper.style.display = "flex"; // Show the loader itself
    }

    // Simulate the shutdown process by waiting for 3 seconds before hiding the loader
    setTimeout(() => {
      // Hide the loader and container after the delay
      if (shutdownContainer && loaderWrapper) {
        shutdownContainer.style.display = "none";
        loaderWrapper.style.display = "none";
      }

      // Update the status message with the simulated response
      const statusMsg = document.getElementById("status-msg");
      if (statusMsg) {
        statusMsg.textContent = "Shutdown initiated... (simulated)";
      }
    }, 3000); // Simulated delay of 3 seconds

    // Optionally, you can send the shutdown POST request to the server
    fetch("/shutdown", {
      method: "POST",
    })
      .then((response) => response.text())
      .then((data) => {
        // Simulate shutdown process after the delay (optional)
      })
      .catch((error) => {
        console.error("Error sending shutdown request:", error);
      });
  };

  // When the user clicks "No" (cancel shutdown)
  document.getElementById("confirmNo").onclick = function () {
    // Hide the confirmation modal
    confirmModal.style.display = "none";
  };
}

// Function to show the snapshot confirmation modal
function showSnapshotModal(event) {
  event.preventDefault(); // Prevent the default action (page navigation)

  // Show the snapshot confirmation modal
  const snapshotModal = document.getElementById("snapshotConfirmModal");
  snapshotModal.style.display = "flex"; // Display the modal

  // When the user clicks "Yes" (confirm the snapshot)
  document.getElementById("confirmYes").onclick = function () {
    snapshotModal.style.display = "none"; // Hide the modal

    console.log("Snapshot confirmed. Proceeding to the next step.");

    // Call the fetchPrediction function and redirect after it's done
    fetchPrediction();
  };

  // When the user clicks "No" (cancel the snapshot)
  document.getElementById("confirmNo").onclick = function () {
    snapshotModal.style.display = "none"; // Close the modal
    console.log("Snapshot not confirmed. Returning to the image.");
  };
}

// Function to show the batch confirmation modal
function showBatchConfirmationModal(event) {
  event.preventDefault(); // Prevent default action (page navigation)

  // Show the confirmation modal
  const batchConfirmModal = document.getElementById("batchConfirmModal");
  batchConfirmModal.style.display = "flex"; // Display the modal

  // When the user clicks "Yes" (confirm the batch is done)
  document.getElementById("confirmYes").onclick = function () {
    batchConfirmModal.style.display = "none"; // Hide the modal

    // Proceed to the next step (you can add further logic here, like navigation)
    console.log("Batch confirmed. Proceeding to the next step.");

    // Redirect to the next page
    var redirectToSlideIn = document
      .getElementById("toSlideIn")
      .getAttribute("data-url");
    window.location.href = redirectToSlideIn;
  };

  // When the user clicks "No" (cancel the batch confirmation)
  document.getElementById("confirmNo").onclick = function () {
    batchConfirmModal.style.display = "none"; // Close the modal
    console.log("Batch not confirmed. Returning to the image.");
  };
}

function showLoadingModal() {
  document.querySelector(".predictionModalText").innerText =
    "Processing prediction...";
  document.querySelector(".predictionModal").style.display = "block";
  document.querySelector(".spinner").style.display = "block";
  document.querySelector(".checkmark").style.display = "none";
  document.querySelector(".close-btn").style.display = "none";
}

let redirectUrl = null;

function showPredictionModal(message, redirect) {
  document.querySelector(".predictionModalText").innerText = message;
  document.querySelector(".spinner").style.display = "none";
  document.querySelector(".checkmark").style.display = "block";
  document.querySelector(".close-btn").style.display = "inline-block";
  redirectUrl = redirect; // Store redirect URL for later
}

function closePredictionModal() {
  document.querySelector(".predictionModal").style.display = "none";

  if (redirectUrl) {
    window.location.href = redirectUrl;
  }
}

// Define fetchPrediction function returning a promise
function fetchPrediction() {
  showLoadingModal();

  return fetch("/predict")
    .then((response) => response.json())
    .then((data) => {
      var redirectToBoundingBox = document
        .getElementById("urlData")
        .getAttribute("data-url");

      if (data.status === "success") {
        showPredictionModal(
          "Prediction completed!",
          redirectToBoundingBox
        );
      } else {
        showPredictionModal("Something went wrong.", null);
      }
    })
    .catch((error) => {
      showPredictionModal("An error occurred.", null);
      console.error(error);
    });
}
