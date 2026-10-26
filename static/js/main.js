// Mobile navigation toggle
const toggle = document.getElementById("nav-toggle");
const links = document.getElementById("nav-links");
if (toggle && links) {
  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
}

// Prediction form: submit with fetch so the page does not reload.
// Without JavaScript the form still works as a normal POST (server renders the result).
const form = document.getElementById("predict-form");
if (form) {
  const resultBody = document.getElementById("result-body");
  const button = document.getElementById("submit-btn");

  const clearErrors = () => {
    form.querySelectorAll(".field").forEach((f) => f.classList.remove("invalid"));
    form.querySelectorAll(".error").forEach((e) => (e.textContent = ""));
  };
  const showErrors = (errors) => {
    Object.entries(errors).forEach(([name, message]) => {
      const box = document.getElementById(`error-${name}`);
      if (box) {
        box.textContent = message;
        box.closest(".field").classList.add("invalid");
      }
    });
  };
  const show = (nodes) => resultBody.replaceChildren(...nodes);
  const paragraph = (text, cls) => {
    const p = document.createElement("p");
    p.textContent = text;
    if (cls) p.className = cls;
    return p;
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearErrors();
    button.disabled = true;
    button.textContent = "Estimating...";
    try {
      const response = await fetch(form.action, {
        method: "POST",
        headers: { Accept: "application/json" },
        body: new FormData(form),
      });
      const data = await response.json();
      if (data.ok) {
        show([paragraph(data.price_formatted, "price"), paragraph(`Model: ${data.model}`, "muted")]);
      } else {
        if (data.errors) showErrors(data.errors);
        show([paragraph(data.message || "Could not estimate a price.", "error")]);
      }
    } catch (err) {
      show([paragraph("Could not reach the server. Please try again.", "error")]);
    } finally {
      button.disabled = false;
      button.textContent = "Estimate price";
    }
  });
}
