# Load necessary libraries
library(shiny)
library(DT)
library(ggplot2)
library(dplyr)

# Set the working directory (adjust the path accordingly)

# Read the CSV data
data_file <- './samples.csv'
data <- read.csv(data_file)
cohorts <- unique(data$cohort)

# Define UI for application that displays the data in a datatable format
ui <- fluidPage(
    titlePanel("Data Exploration"),
    sidebarLayout(
        sidebarPanel(
            # Application title
            # multiple selection of the cohort
            selectInput("cohort", "Select cohort:", choices = cohorts, multiple = TRUE, selected = cohorts[1]),
            # select if you want metagenomics or metatranscriptomics
            selectInput("type", "Select type:", choices = c("metagenomics", "metabolomics", "both")),

        ),
        mainPanel(
            tableOutput("phenotype_table"),
            # plot the histogram of the age distribution
            plotOutput("age_dist"),
            plotOutput("sex_dist"),
            # Display the data in a datatable
            DTOutput("data_table")
        )
    )
)

# Define server logic to render the datatable
server <- function(input, output) {
    # set the table to be reactive
    filtered_data <- reactive({
        # Filter the data based on the selected cohort
        cur_data <- data[data$cohort %in% input$cohort, ]
        # Filter the data based on the selected type
        if (input$type == "metagenomics") {
            cur_data <- cur_data[cur_data$metagenomic_data == "Y", ]
        } else if (input$type == "metabolomics") {
            cur_data <- cur_data[cur_data$metabolomic_data == "Y", ]
        } else if (input$type == "both") {
            cur_data <- cur_data[cur_data$metagenomic_data == "Y" & cur_data$metabolomic_data == "Y", ]
        }
        return(cur_data)
    })

    output$data_table <- renderDT({
        datatable(filtered_data())
    })

    # render a table of phenotype data
    output$phenotype_table <- renderTable({
        req(filtered_data())  # Ensure the filtered data is available
        # tratify the data by cohort
        cohorts <- unique(filtered_data()$cohort)
        pheno_tb <- list()
        pheno_tb_ind <- 1
        for (cohort in cohorts) {
            cur_data <- filtered_data()[filtered_data()$cohort == cohort, ]
            pheno_tb[[pheno_tb_ind]] <- table(cur_data$phenotype)
            pheno_tb_ind <- pheno_tb_ind + 1
        }
        # merge the tables
        pheno_tb <- do.call(rbind, pheno_tb)
        # add a row of sum
        pheno_tb <- rbind(pheno_tb, colSums(pheno_tb, na.rm = TRUE))
        rownames(pheno_tb) <- c(cohorts, "total")
        return(pheno_tb)
        
    }, , rownames = TRUE)

    # plot the age distribution
    output$age_dist <- renderPlot({
    req(filtered_data())  # Ensure the filtered data is available
    # Plot the age distribution stratified by cohort with overlay
    ggplot(filtered_data(), aes(x = age, fill = cohort)) + 
        geom_histogram(binwidth = 5, position = "identity", color = "black", alpha = 0.3) +
        labs(title = "Age Distribution Stratified by Cohort", x = "Age", y = "Frequency") +
        theme_minimal()
    })

    # plot the sex distribution
    output$sex_dist <- renderPlot({
    req(filtered_data())  # Ensure the filtered data is available
    # Plot the sex distribution stratified by cohort
    ggplot(filtered_data(), aes(x = sex, fill = cohort)) +  # Assuming 'sex' is the column name for sex in your data
        geom_bar(position = "dodge", color = "black") +
        labs(title = "Sex Distribution Stratified by Cohort", x = "Sex", y = "Count") +
        theme_minimal()
})
}

# Run the application
shinyApp(ui = ui, server = server)