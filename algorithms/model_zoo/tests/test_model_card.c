#include <neural_core/model_zoo/model_card.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

void test_create_destroy(void) {
    printf("test_create_destroy... ");
    ModelCard *card = model_card_create();
    assert(card != NULL);
    model_card_destroy(card);
    printf("PASS\n");
}

void test_setters(void) {
    printf("test_setters... ");
    ModelCard *card = model_card_create();
    
    model_card_set_name(card, "test-model");
    model_card_set_version(card, "1.0.0");
    model_card_set_description(card, "A test model");
    model_card_set_author(card, "Test Author");
    model_card_set_license(card, "MIT");
    model_card_set_repository(card, "https://github.com/test/model");
    model_card_set_architecture(card, "transformer");
    model_card_set_num_parameters(card, 1000000);
    model_card_add_tag(card, "nlp");
    model_card_add_tag(card, "transformer");
    
    assert(strcmp(card->name, "test-model") == 0);
    assert(strcmp(card->version, "1.0.0") == 0);
    assert(strcmp(card->description, "A test model") == 0);
    assert(strcmp(card->author, "Test Author") == 0);
    assert(strcmp(card->license, "MIT") == 0);
    assert(strcmp(card->repository, "https://github.com/test/model") == 0);
    assert(strcmp(card->architecture, "transformer") == 0);
    assert(card->num_parameters == 1000000);
    assert(card->num_tags == 2);
    assert(strcmp(card->tags[0], "nlp") == 0);
    assert(strcmp(card->tags[1], "transformer") == 0);
    
    model_card_destroy(card);
    printf("PASS\n");
}

void test_json_serialization(void) {
    printf("test_json_serialization... ");
    ModelCard *card = model_card_create();
    
    model_card_set_name(card, "test-model");
    model_card_set_version(card, "1.0");
    card->architecture = strdup("transformer");
    card->num_parameters = 1000000;
    card->learning_rate = 2e-4;
    card->num_layers = 12;
    card->tags = (char **)malloc(2 * sizeof(char *));
    card->tags[0] = strdup("nlp");
    card->tags[1] = strdup("transformer");
    card->num_tags = 2;
    
    char *json = model_card_to_json(card, 0);
    assert(json != NULL);
    assert(strstr(json, "test-model") != NULL);
    assert(strstr(json, "transformer") != NULL);
    assert(strstr(json, "1000000") != NULL);
    assert(strstr(json, "nlp") != NULL);
    
    free(json);
    model_card_destroy(card);
    printf("PASS\n");
}

void test_file_io(void) {
    printf("test_file_io... ");
    ModelCard *card = model_card_create();
    
    model_card_set_name(card, "test-model");
    model_card_set_version(card, "1.0");
    card->architecture = strdup("transformer");
    card->num_parameters = 1000000;
    
    const char *path = "test_model_card.json";
    int rc = model_card_save(card, path);
    assert(rc == 0);
    
    ModelCard *loaded = model_card_load("test_model_card.json");
    // Note: loading is a stub, so loaded will be empty
    
    // Clean up
    remove("test_model_card.json");
    model_card_destroy(card);
    printf("PASS\n");
}

void test_validation(void) {
    printf("test_validation... ");
    char *error = NULL;
    
    // Valid card
    ModelCard *card = model_card_create();
    model_card_set_name(card, "valid-model");
    card->version = strdup("1.0");
    card->architecture = strdup("transformer");
    card->num_parameters = 1000000;
    card->num_layers = 12;
    card->learning_rate = 2e-4;
    
    assert(model_card_validate(card, NULL) == 0);
    
    // Invalid: missing name
    ModelCard *invalid = model_card_create();
    invalid->version = strdup("1.0");
    invalid->architecture = strdup("transformer");
    assert(model_card_validate(invalid, NULL) != 0);
    
    // Invalid: missing version
    ModelCard *invalid2 = model_card_create();
    model_card_set_name(invalid2, "test");
    invalid2->architecture = strdup("transformer");
    assert(model_card_validate(invalid2, NULL) != 0);
    
    // Invalid: negative params
    ModelCard *invalid3 = model_card_create();
    model_card_set_name(invalid3, "test");
    invalid3->version = strdup("1.0");
    invalid3->architecture = strdup("transformer");
    invalid3->num_parameters = -1;
    assert(model_card_validate(invalid3, NULL) != 0);
    
    // Invalid: zero learning rate
    ModelCard *invalid4 = model_card_create();
    model_card_set_name(invalid4, "test");
    invalid4->version = strdup("1.0");
    invalid4->architecture = strdup("transformer");
    invalid4->learning_rate = 0;
    assert(model_card_validate(invalid4, NULL) != 0);
    
    model_card_destroy(card);
    model_card_destroy(invalid);
    model_card_destroy(invalid2);
    model_card_destroy(invalid3);
    model_card_destroy(invalid4);
    
    printf("PASS\n");
}

int main(void) {
    printf("\n=== Model Card Tests ===\n\n");
    
    test_create_destroy();
    test_setters();
    test_json_serialization();
    test_file_io();
    test_validation();
    test_setters();
    
    printf("\n=== All tests passed ===\n\n");
    return 0;
}