#' Load and install R packages.
#'
#' This function checks for the given packages in the current R installation.
#' If a package is not found, it installs it from the specified repository.
#' It then loads the packages into the current R session.
#'
#' @param package A character vector of package names.
#' @param repos The repository URL to install packages from.
#'        Default is the Goettingen (Germany) mirror.
#'
#' @return A logical vector indicating successful loading of packages.
load.package <- function(package, repos, lib) {

  if (missing(lib)) {
    lib <- .libPaths()[1]
  }

  # list of packages missing
  pkgs <- installed.packages(
    lib.loc = lib
  )
  missing <- package[!package %in% pkgs[, 'Package']]

  # check wich packages are not intalled and install them
  if (!is.null(missing)) {
    if (missing(repos)) {
      # use Erlangen (Germany) mirror as default
      repos <- 'https://ftp.fau.de/cran/'
    }
    install.packages(
      missing,
      lib = lib,
      dependencies = TRUE,
      repos = repos
    )
  }

  # load all packages
  sapply(package, require,
         character.only = TRUE ,
         lib.loc = lib
  )
}

#' Create APA-styled HTML table using the gt package.
#'
#' This function uses the gt package to create an APA-styled table with the
#' specified appearance.
#'
#' @param x A data frame or table to be styled.
#' @param title A character string specifying the title of the table.
#'        Default is an empty space.
#'
#' @param stub A logical value to determine if row names should be used as stub.
#'        Default is TRUE.
#'
#' @return A gt object with the specified stylings.
apa <- function(x, title = " ", stub = T) {
  # get gt package for making html tables
  load.package('gt')

  gt(x, rownames_to_stub = stub) %>%
    tab_stubhead(label = "Predictor") %>%
    tab_options(
      table.border.top.color = "white",
      heading.title.font.size = px(16),
      column_labels.border.top.width = 3,
      column_labels.border.top.color = "black",
      column_labels.border.bottom.width = 3,
      column_labels.border.bottom.color = "black",
      stub.border.color = "white",
      table_body.border.bottom.color = "black",
      table.border.bottom.color = "white",
      table.width = pct(100),
      table.background.color = "white"
    ) %>%
    cols_align(align="center") %>%
    tab_style(
      style = list(
        cell_borders(
          sides = c("top", "bottom"),
          color = "white",
          weight = px(1)
        ),
        cell_text(
          align="center"
        ),
        cell_fill(color = "white", alpha = NULL)
      ),
      locations = cells_body(
        columns = everything(),
        rows = everything()
      )
    ) %>%
    #title setup
    tab_header(
      title = html("<i>", title, "</i>")
    ) %>%
    opt_align_table_header(align = "left")
}

#' Format values for presentation.
#'
#' This function formats the given value to be presented in reports or tables.
#' If the absolute value is less than 0.001, it returns '< 0.001'. Otherwise,
#' it rounds and formats the value according to the given parameters.
#'
#' @param value A numeric value to be formatted.
#' @param nsmall A non-negative integer giving the minimum number of digits to
#'        the right of the decimal point. Default is 3.
#'
#' @param simplify A logical value. If TRUE, removes the '<' and '= ' prefixes.
#'        Default is FALSE.
#'
#' @return A character string of the formatted value.
format.value <- function(value, nsmall = 3, simplify = FALSE) {

  if (abs(value) < 0.001) {
    print_value <- '< 0.001'
  } else {
    print_value <- paste0('= ' , format(round(value, digits = nsmall), nsmall = nsmall))
  }

  if (simplify) {
    print_value <- gsub('< |= ', '', print_value)
  }

  return(print_value)
}