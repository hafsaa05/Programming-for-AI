score <- as.numeric(readline("Enter score (0-100): "))

if (score >= 90) {
  grade <- "A"
} else if (score >= 80) {
  grade <- "B"
} else if (score >= 70) {
  grade <- "C"
} else if (score >= 0) {
  grade <- "Fail"
} else {
  grade <- "Invalid input"
}

cat("The grade is", grade, "\n")

