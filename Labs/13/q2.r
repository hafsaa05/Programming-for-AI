sum <- function(values) {
  return(sum(values[values %% 2 == 0]))
}

values <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
result <- sum(values)
cat("Sum of even numbers:", result, "\n")
