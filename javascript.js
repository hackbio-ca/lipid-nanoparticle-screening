document.addEventListener("DOMContentLoaded", () => {
  const button = document.querySelector("button");
  const numberDiv = document.querySelector(".number");

  function getRandomNumber() {
    const num = Math.random() * (8 - 5) + 5; // between 5 and 8
    return num.toFixed(2); // 2 decimals
  }

  button.addEventListener("click", () => {
    numberDiv.textContent = getRandomNumber();
  });
});