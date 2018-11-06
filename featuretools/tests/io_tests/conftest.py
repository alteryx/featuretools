import pytest
import sqlalchemy as sa


def create_widgets_table(engine, metadata):
    sa.Table('widgets', metadata,
             sa.Column('my_id', sa.Integer, primary_key=True),
             sa.Column('name', sa.String),
             sa.Column('quantity', sa.Integer),
             sa.Column('price', sa.Float))
    metadata.create_all(engine)


def create_factories_table(engine, metadata):
    sa.Table('factories', metadata,
             sa.Column('id', sa.Integer, primary_key=True),
             sa.Column('city', sa.String),
             sa.Column('widget_produced', sa.ForeignKey("widgets.my_id")))
    metadata.create_all(engine)


def create_customers_tables(engine, metadata):
    sa.Table('customers', metadata,
             sa.Column('name', sa.String),
             sa.Column('id', sa.Integer, primary_key=True))

    sa.Table('customers_widgets', metadata,
             sa.Column('customer_id', sa.Integer, sa.ForeignKey("customers.id")),
             sa.Column('widget_id', sa.Integer, sa.ForeignKey("widgets.my_id")))

    metadata.create_all(engine)


def create_composite_primary_key_tabel(engine, metadata):
    sa.Table('people', metadata,
             sa.Column('id', sa.Integer, primary_key=True),
             sa.Column('state', sa.Integer, primary_key=True))
    metadata.create_all(engine)


@pytest.fixture
def sqlite():
    engine = sa.create_engine('sqlite:///:memory:', echo=False)
    metadata = sa.MetaData()
    connection = engine.connect()

    create_widgets_table(engine, metadata)
    create_factories_table(engine, metadata)
    create_customers_tables(engine, metadata)

    widgets = sa.Table('widgets', metadata, autoload=True)
    factories = sa.Table('factories', metadata, autoload=True)
    customers = sa.Table('customers', metadata, autoload=True)
    customers_widgets = sa.Table('customers_widgets', metadata, autoload=True)

    insert_widget = widgets.insert()
    widget_record = connection.execute(insert_widget, name="whatsit", quantity="8", price="0.57")
    widget_id = widget_record.inserted_primary_key[0]

    insert_factory = factories.insert()
    connection.execute(insert_factory, city="Boston", widget_produced=widget_id)

    insert_customer = customers.insert()
    customer_record = connection.execute(insert_customer, name="Andrew Ng")
    customer_id = customer_record.inserted_primary_key[0]

    insert_customers_widget = customers_widgets.insert()
    connection.execute(insert_customers_widget, customer_id=customer_id, widget_id=widget_id)

    return engine.connect()


@pytest.fixture
def sqlite_composite_pk():
    engine = sa.create_engine('sqlite:///:memory:', echo=False)
    metadata = sa.MetaData()

    create_composite_primary_key_tabel(engine, metadata)
    return engine.connect()
