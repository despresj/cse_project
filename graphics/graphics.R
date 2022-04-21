library(tidyverse)
library(jtools)
library(extrafont)

accuracy <- read_csv("graphics/graphics_data/accuracy_table.csv", skip = 1) %>% 
  mutate_if(is.numeric, round, 4) %>% 
  select(-1) %>% 
  t() 

rmse <- accuracy[, 3:4]
colnames(rmse) <- c( "RMSE Classifier",  "RMSE Regressor")
rmse %>% 
kbl( caption = "Model Performance", booktabs = T, "latex") %>% 
  kable_styling(latex_options = c("striped", "hold_position")) %>%
  add_footnote("Note: this is the full dataset with\n 6497 observations") %>% 
  clipr::write_clip()



colnames(rmse) <- c( "RMSE Chemical Data",  "RMSE Vivino Data")
rmse %>% 
kbl(booktabs = T) %>% 
  kable_material_dark("striped", font_size = 22) %>% 
  row_spec(0, color = "red") %>% 
  add_footnote("Comparing the degree to which wine quality can be modeled given features avalible", notation = "none") 
accuracy <- accuracy[, 1:2]

colnames(accuracy) <- c("Accuracy", "AUC")
library(kableExtra)
accuracy %>% 
  kbl( caption = "Model Performance", booktabs = T, "latex", align = c("c", "c")) %>% 
  kable_styling(latex_options = c("striped", "hold_position")) %>%
  clipr::write_clip()


cbind(row.names(accuracy), c(0.8734, 0.8441, 0.8704, 0.880)) %>% 
  kbl( caption = "Model Performance", booktabs = T, "latex") %>% 
  kable_styling(latex_options = c("striped", "hold_position")) %>%
  add_footnote("Note: this is the full dataset with\n 6497 observations") %>% 
  clipr::write_clip()


  
dark_theme_gray <- read_csv("graphics/graphics_data/vivino.csv") %>% 
  select(-1)

vivino %>% 
  ggplot(aes(x = Rating * 2)) +
  geom_histogram(bins = 30, color = "black") +
  labs(x = "Wine Rating", y = NULL) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(label = scales::comma) +
  theme_apa() 
ggsave("docs/latex_files/plots/vivino_rating.png", width = 2.8, height = 3)

vivino_kable <- vivino %>% 
  select_if(is.numeric) %>% 
  map_df(~map(list(Mean = mean, Std = sd), exec, .x), .id = "Feature") %>% 
  mutate_if(is.numeric, round, 3)

vivino_kable %>%   
  kbl( caption = "Vivino Features table", booktabs = T, "latex") %>% 
  kable_styling(latex_options = c("striped", "hold_position")) %>%
  clipr::write_clip()

vivino_kable %>% 
  kbl(booktabs = T) %>% 
  kable_material_dark("striped", font_size = 22) %>% 
  row_spec(0, color = "red") %>% 
  add_footnote("Note: this is the full dataset with 13,564 observations", notation = "none") 


cat_preds <- vivino %>% 
  select(Country, Region, Winery, wine_type) %>% 
  
  map_df(~map(list(Distinct = n_distinct), exec, .x), .id = "Feature") %>% 
  mutate_if(is.numeric, round, 3) %>% 
  arrange(Distinct)
cat_preds %>% 
  kbl( caption = "Vivino Features table", booktabs = T, "latex") %>% 
  kable_styling(latex_options = c("striped", "hold_position")) %>%
  clipr::write_clip()

cat_preds %>% 
  mutate(Feature = str_replace_all(Feature,"wine_type", "Wine Type"))
  kbl(booktabs = T) %>% 
  kable_material_dark("striped", font_size = 22) %>% 
  row_spec(0, color = "red")

df <- read_csv("graphics/graphics_data/transformations_log.csv") %>% 
  select(-1) %>% 
  rename(`No Transformation` = baseline_results,
         `Dropped Outliers` = robscale_results,
         `Min Max Scale` = min_max_results,
         `Rescaled to Uniform` = quant_transfmr_results,
         `Standard Scale` = std_scale_results,
         `Yao Johnson Transform` = pwr_transform_results)

training <- read_csv("graphics/graphics_data/training_set.csv") %>% 
  select(-is_red, -quality)



summary_table <- read_csv("data/processed/wine_data_combined.csv") %>% 
  rename_with(~str_to_title(str_replace_all(.x, "_", " "))) %>% 
  map_df(~ list(Mean = mean, Std = sd) %>%
           map(exec, .x), .id = "Feature") %>%
  mutate_if(is.numeric, round, 3)

summary_table %>% 
  filter(Feature != "Quality") %>% 
  kbl(booktabs = T) %>% 
  kable_material_dark("striped", font_size = 22) %>% 
  row_spec(0, color = "red") %>% 
  add_footnote("Note: this is the full dataset with 6,497 observationss", notation = "none") 
  
df_raw %>% 
 map(list(mean = mean, sd = sd)) 
  
tibble(feature = str_to_title(str_replace_all(colnames(training), "_", " ")), 
       Description = rep("             ", length(colnames(training)))) %>% 


par(mfrow=c(1,2))
hist(training$residual_sugar, 100)
hist((training$residual_sugar - mean(training$residual_sugar))/sd(training$residual_sugar), 100)
# transform data to stand dev. from mean.


df_summary <- df %>% 
  pivot_longer(cols =  everything(), names_to = "scaling_method", values_to = "auc") 
  group_by(scaling_method) %>% 
  summarise(mean = mean(auc), sd = sd(auc), med = median(auc)) %>% 
  mutate_if(is.numeric, round, digits = 3) %>% 
  arrange(mean)


library(kableExtra)
df_summary %>% 
  rename_with(~str_to_title(str_replace(.x, "_", " "))) %>% 
  rename(`Standard Dev` = Sd, Median = Med) %>% 
  kbl( caption = "Demo table", booktabs = T, "latex") %>% 
  kable_styling(latex_options = c("striped", "hold_position")) %>% 
  clipr::write_clip()

df %>% 
  mutate(scaling_method = factor(scaling_method, pull(df_summary, scaling_method))) %>% 
  ggplot(aes(x = auc)) +
  geom_histogram(bins = 100, fill = "#f9b642", color = "#253582", alpha = 0.4) +
  geom_vline(xintercept = mean(df$auc), linetype = "dashed") +
  facet_wrap(~scaling_method, ncol = 3) +
  labs(x = "Area under the ROC Curve", y = "Samples") +
  theme_minimal() 

# ggsave("graphics/plots/auc_comparasent.png", width = 16, height = 9)

ordered_names <-  df %>% 
  summarise(across(.fns = mean)) %>% 
  pivot_longer(cols = everything()) %>% 
  arrange((value)) %>% 
  pull(name)

mean_auc <- df %>% 
  pivot_longer(cols = everything(), names_to = "scaling_method", values_to = "auc") %>% 
  summarise(mean_auc = mean(auc)) %>% 
  pull(mean_auc)

df %>% 
  pivot_longer(cols = everything(), names_to = "scaling_method", values_to = "auc") %>% 
   mutate(scaling_method = factor(scaling_method, ordered_names)) %>% 
   ggplot(aes(x = auc)) +
   geom_histogram(bins = 100, color = "black", fill = "black") +
   geom_vline(xintercept = mean_auc, linetype = "dashed") +
   facet_wrap(~scaling_method, ncol = 1) +
   labs(x = "Area under the ROC Curve (AUC)", y = NULL) +
   theme_apa() 
  

# ggsave("docs/latex_files/plots/auc_comparasent.png", width = 2.8, height = 6)

training %>% 
  map_df(~(.x - mean(.x))/ sd(.x)) %>% 
  rename_with(~paste0(.x, "_\ntransformed")) %>% 
  bind_cols(training) %>%
  pivot_longer(cols =  everything()) %>% 
  mutate(name = str_replace_all(name, "_", " "),
         name = str_to_title(name),
         Transformed = if_else(str_detect(name, "Transform"), "Post Transfom", "Pre-Transform")
         ) %>% 
  ggplot(aes(x = value, fill = Transformed)) +
  geom_histogram(bins = 50) +
  facet_wrap(~name, scales = "free", ncol = 4) +
  scale_fill_manual(values = c("grey12", "grey69"))+
  labs(color = "", x = "Value", y = "Count") +
  theme_bw(base_size = 20) +
  theme_apa() +
  theme(text=element_text(family="Times New Roman", size = 12)) +
  theme(legend.position = "bottom")

# ggsave("docs/latex_files/plots/transformations.png", width = 7.5, height = 8)

plot_data <- read_csv("data/processed/wine_data_combined.csv") %>%
  group_by(quality) %>% 
  summarise(n = n())
library(scales)


plot_data %>% 
  ggplot(aes(x = as.character(quality), y=n)) +
  geom_col(fill="grey52", color="black") +
  geom_text(aes(label=n), nudge_y = 58, size=2) +
  scale_fill_manual(values = c("grey12"))+
  labs(color = "", x = "Wine Quality", y = "Count") +
  theme_bw(base_size = 20) +
  theme_apa() +
  scale_x_discrete(breaks = 3:9) +
  scale_y_continuous(labels = comma) +
  theme(text=element_text(family="Times New Roman", size = 12)) +
  theme(legend.position = "bottom") 

# ggsave("docs/latex_files/plots/target.png", width = 3, height = 3)

library(ggdark)

plot_data %>% 
  ggplot(aes(x = as.character(quality), y=n)) +
  geom_col(fill="red", color="white") +
  geom_text(aes(label=n), nudge_y = 58, size=5) +
  scale_fill_manual(values = c("grey12"))+
  labs(color = "", x = "Wine Quality", y = "Count") +
  dark_theme_gray(base_size=15)+
  scale_x_discrete(breaks = 3:9) +
  scale_y_continuous(labels = comma) +
  theme(legend.position = "bottom") 



# ggsave("docs/latex_files/plots/target.png", width = 3, height = 3)

vivino %>% 
  mutate(Rating = Rating * 2) %>% 
  ggplot(aes(x = Rating)) +
  geom_histogram(bins = 25, color = "white", fill = "red") +
  labs(color = "", x = "Wine Quality", y = "Count") +
  dark_theme_gray(base_size=15)+
  scale_x_continuous(breaks = 0:10) +
  scale_y_continuous(labels = comma, breaks = seq(0, 5000, 500)) +
  theme(legend.position = "bottom") 
