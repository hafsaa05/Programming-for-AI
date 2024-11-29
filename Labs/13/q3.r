LeapYear <- function(year) {
  (year %% 4 == 0 && year %% 100 != 0) || (year %% 400 == 0)
}

year <- as.integer(readline("Enter a year: "))

if (!is.na(year) && year > 0) {
  if (LeapYear(year)) {
    cat(paste(year, "is a leap year.\n"))
  } else {
    cat(paste(year, "is not a leap year.\n"))
  }
} else {
  cat("Invalid input. Please enter a positive integer.\n")
}
